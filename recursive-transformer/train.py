from model import *
from dataloaders import *
from utils import *
from optimizer import *
import time

def train(model, train_loader, optimizer):
    total_batch_size = train_loader.B * train_loader.T #2**19 tokens
    max_steps = 50
    grad_accum_steps = total_batch_size // (train_loader.B * train_loader.T)
    assert total_batch_size % (B * T) == 0
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


def setup_session_config():
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    torch.set_float32_matmul_precision('high')


if __name__== "__main__":

    device = autodetect_device()
    print(f"Running on {device}")
    setup_session_config()

    B = 16
    T = 512
    train_loader = DataLoaderLite(B=B, T=T)
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    model = torch.compile(model, backend='inductor')

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
    train(model, train_loader, optimizer)

