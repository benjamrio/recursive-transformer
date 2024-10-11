from model import *
from dataloaders import *
from utils import *
from optimizer import *
import time
import os


def autodetect_device():
    device = 'cpu'
    if torch.cuda.is_available():
        print(f'{torch.cuda.device_count()} cuda devices available')
        device = 'cuda:1'
    return(device)


def setup_session_config():
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    torch.set_float32_matmul_precision('high')


if __name__== "__main__":

    device = autodetect_device()
    print(f"Running on {device}")
    setup_session_config()

    B = 4
    T_prst = 1
    total_batch_size = B * T_prst

    dataset = XorDataset(n=10002)
    train_loader = DatasetLoader(B=B, T=T_prst, dataset=dataset, sampler='sequential')
    vocab_size = 2
    config = GPTConfig(vocab_size=vocab_size, block_size=4, d_emb=16, n_head=2, n_layer=8)
    model = GPT(config)
    model.to(device)
    #model = torch.compile(model, backend='inductor')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) #model.configure_optimizers(weight_decay=0.1, learning_rate=5, device=device)

    max_steps = 500
    grad_accum_steps = total_batch_size // (train_loader.B * train_loader.T)
    assert total_batch_size % (B * T_prst) == 0

    T_past = 2
    past_embs = torch.full((B, T_past, config.d_emb), 0).to(device)

    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0

        x, y = next(train_loader)
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss, embs = model(x, past_embs, y)
        # loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

        past_embs = embs[:, :T_past, :].detach() # ! should we propagate one time through graph??
        print(past_embs)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #lr = get_lr(step, max_steps)
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

    '''
    save_dir = "outputs"
    final_save_path = os.path.join(save_dir, "model_final.pt")
    torch.save({
        'step': max_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_accum.item(),
    }, final_save_path)
    print(f"Final model saved to {final_save_path}")
    '''