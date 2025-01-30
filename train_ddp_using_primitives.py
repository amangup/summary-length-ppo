from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import re
import math
import time
import sys

import spacy
nlp = spacy.load("en_core_web_sm")

import matplotlib.pyplot as plt

dataset_id = 'amang1802/summary_train_med'
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

pattern = re.compile(r'Summarize the text using exactly (\d+) words:')

os.environ['NCCL_DEBUG']='INFO'


def parse_target_words(query):
    match = re.search(pattern, query)
    return int(match.group(1))


def word_count(text):
    doc = nlp(text)
    return sum(1 for token in doc if token.is_alpha)


def get_reward(queries_batch, response_batch, range_norm=True):
    queries = torch.tensor([q['input_ids'][0] for q in queries_batch])
    responses = torch.tensor(response_batch)
    
    query_strs = tokenizer.batch_decode(queries)
    target_word_lengths = torch.tensor([parse_target_words(query)
                                        for query in query_strs]).float()
    
    response_strs = tokenizer.batch_decode(responses, skip_special_tokens=True)
    word_lengths = torch.tensor([word_count(text)
                                 for text in response_strs]).float()

    rewards = target_word_lengths - word_lengths
    if range_norm:
        rewards = torch.exp(0.7 * -torch.abs(rewards) + torch.log(torch.tensor(2))) - 1
    
    return {"reward": rewards.tolist()}


class StopEoT(StoppingCriteria):
    def __init__(self, stop_id=128009):
        self.stop_token = '<|eot_id|>'

    def __call__(self, input_ids, scores, **kwargs):
            return self.stop_token == tokenizer.decode(input_ids[-1])


stopping_criteria = StoppingCriteriaList([StopEoT()])


def generate_responses(model, device, queries_batch):
    inputs, masks = zip(*[(query['input_ids'][0], query['attention_mask'][0])
                          for query in queries_batch])
    
    inputs = torch.tensor(list(inputs))
    masks = torch.tensor(list(masks))
    
    outputs = model.generate(inputs=inputs.to(device),
                             attention_mask=masks.to(device),
                             max_new_tokens=64,
                             do_sample=False,
                             top_p=None,
                             temperature=None,
                             stopping_criteria=stopping_criteria
                            )

    response_only = outputs[:, inputs.shape[1]:]
    
    return {"generated_tokens": list(torch.unbind(response_only, dim=0))}


def _get_partition_range(ds, rank, size):
    ds_len = ds.num_rows    
    if rank < size - 1:
        partition_range = (ds_len//size * rank, ds_len//size * (rank+1))
    else:
        partition_range = (ds_len//size * rank, ds_len)

    return partition_range


def get_data_shard(rank, size):
    summary_train_ds = load_dataset(dataset_id)
    train_range = _get_partition_range(summary_train_ds['train'], rank, size)
        
    return summary_train_ds['train'].select(range(*train_range)), summary_train_ds['test']
    

# Output these tensors
# X, with shape (batch_size, 1024). These are the inputs.
# Y, with shape (batch_size, 1024). This is almost exactly the same as X,
#    only shifted by 1
# R, with shape (batch_size, 1024). These are the rewards for each token. 
#    We mask out the tokens we don't want to reward.
def dataloader(model, device, ds, start, end):
    batch = ds.select(range(start, end))
    batch = batch.map(lambda batch: generate_responses(model, device, batch), batch_size=512, batched=True, 
                      input_columns=['tokens'])
    batch = batch.map(get_reward, batched=True, batch_size=512,
                      input_columns=['tokens', 'generated_tokens'])
    
    batch_size = end - start
    inputs = torch.tensor([t['input_ids'][0] for t in batch['tokens']])
    outputs = torch.tensor(batch['generated_tokens'])
    #print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    rewards = torch.tensor(batch['reward'])

    X = torch.cat([inputs, outputs[:, :-1]], dim=1)
    Y = torch.cat([inputs[:, 1:], outputs], dim=1)

    # we don't want to train on input - so the input tokens are masked out
    labels_mask = torch.cat([torch.zeros_like(inputs),
                             torch.ones((batch_size, outputs.shape[1]-1))], dim=1)
    # hacky way to mask out the token used for padding
    labels_mask = labels_mask * (Y != 128001).int()
    
    R = (torch.ones_like(Y) * rewards.unsqueeze(dim=1)) * labels_mask

    # I just like the number 1024 - since inputs.shape[1] is 1024, 
    # and I know there is enough padding in the input,
    # I make the X.shape[1] and Y.shape[1] also 1024. 
    # It also supposedly makes matrix multiplication faster in nvidia GPUs!
    return X[:, outputs.shape[1]-1:], Y[:, outputs.shape[1]-1:], R[:, outputs.shape[1]-1:]


def eval_reward_metrics(model, device, eval_ds):
    responses = eval_ds.map(lambda batch: generate_responses(model, device, batch),
                            batch_size=512, batched=True, input_columns=['tokens'])
    rewards = responses.map(lambda c1, c2: get_reward(c1, c2, False), batched=True, batch_size=512,
                            input_columns=['tokens', 'generated_tokens'])

    rewards_t = torch.tensor(rewards['reward'])
    return rewards_t.mean(), rewards_t.std()
    

def forward(model, device, X, Y, R):
    X, Y, R = X.to(device), Y.to(device), R.to(device)
    output = model(X)
    # output.logits have the shape (batch_size, input_length, vocab_size)
    # logprobs = output.logits.log_softmax(dim=2)
    probs = output.logits.softmax(dim=2)
    P_y_x = probs.gather(2, Y.unsqueeze(2)).squeeze()

    # We want a positive loss for descent. P_y_x elements are negative.
    # If reward is negative P_y_x*R is positive.
    loss = -(P_y_x * R).sum() / P_y_x.sum()
    #loss = (P_y_x * R).mean()
        
    return loss


def training_loop(rank, world_size, loss_queue, eval_queue):
    device = torch.device(f"cuda:{rank}")

    train_ds, eval_ds = get_data_shard(rank, world_size)
    eval_ds = eval_ds.select(range(0, 256))
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map={"": rank})
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    num_epochs = 1
    batch_size = 16
    gradient_accu_steps = 1
    eval_interval = 8
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    
    for epoch in range(num_epochs):
        accu_loss = torch.zeros(1).to(device)
        for step, index in enumerate(range(0, train_ds.num_rows, batch_size)):
            if rank == 0:
                print(f"Step: {(train_ds.num_rows//(batch_size*world_size)) * epoch + step}")
            
            X, Y, R = dataloader(model, device, train_ds, index, index+batch_size)
            loss = forward(model, device, X, Y, R) / gradient_accu_steps
            accu_loss += loss
            loss.backward()
            
            if (step+1) % gradient_accu_steps == 0:
                for param in model.parameters():
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size

                dist.all_reduce(accu_loss, op=dist.ReduceOp.SUM)
                accu_loss /= world_size
                if rank == 0:
                    print(f"Loss: {accu_loss.item()}")

                    effective_step = (step+1) // gradient_accu_steps
                    loss_queue.put((effective_step, accu_loss.item()))

                    if effective_step % eval_interval == 0:
                        print(f"Running eval...")
                        ldiff_mean, ldiff_std = eval_reward_metrics(model, device, eval_ds)
                        print(f"Eval results -- Length diff mean: {ldiff_mean}, Length diff std dev: {ldiff_std}")

                        eval_queue.put((effective_step, ldiff_mean, ldiff_std))


                optimizer.step()
                optimizer.zero_grad()

                accu_loss *= 0

                dist.barrier()


    if rank == 0:
        output_model_id = 'Llama3.2-1B-summary-length-exp7.1'
        model.push_to_hub(output_model_id)
        tokenizer.push_to_hub(output_model_id)
        print("Model pushed to HF hub")


def init_process(rank, size, loss_queue, eval_queue, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, loss_queue, eval_queue)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    world_size = torch.cuda.device_count()
    start_time = time.perf_counter_ns()
    processes = []
    loss_queue = mp.SimpleQueue()
    eval_queue = mp.SimpleQueue()
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, loss_queue, eval_queue, training_loop))
        p.start()
        processes.append(p)

    loss_data = []
    eval_data = []
    try:
        while any(p.is_alive() for p in processes):
            while not eval_queue.empty():
                eval_data.append(eval_queue.get())
                
            while not loss_queue.empty():
                loss_data.append(loss_queue.get())
            
            time.sleep(1.0)
    except: # handle exceptions
        for process in processes:
           if process.is_alive():
               print(f"Terminating process {process.pid}")
               process.terminate()
    finally:
        for process in processes:
            process.join()
        
               
    print(f"Performance: {(time.perf_counter_ns() - start_time) / 1e9:.5f}")

    if loss_data:
        loss_data.sort(key=lambda x: x[0])
        eval_data.sort(key=lambda x: x[0])
        steps, loss = zip(*loss_data)

        plt.plot(list(steps), list(loss))
        plt.savefig('training_loss.png')
        plt.close()

        steps, ldiff_mean, ldiff_std = zip(*eval_data)
        plt.plot(list(steps), list(ldiff_mean), label="Ldiff_mean")
        plt.plot(list(steps), list(ldiff_std), label="Ldiff_std")
        plt.savefig('eval.png')
        plt.close()
