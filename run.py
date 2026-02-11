from pathlib import Path
import os

import fire
import torch
import unisal

# Global list to store captured RNN features for one video
rnn_feats = []

# Directory to save RNN latent vectors
LATENT_SAVE_DIR = Path("rnn_latents")
LATENT_SAVE_DIR.mkdir(exist_ok=True, parents=True)

# Hook function to capture RNN outputs
def hook_fn(module, input, output):
    # output is a tuple: (rnn_feat_seq, hidden)
    rnn_feat_seq = output[0]  # extract the tensor
    rnn_feats.append(rnn_feat_seq.detach().cpu())


def train(eval_sources=('DHF1K', 'SALICON', 'UCFSports', 'Hollywood'),
          **kwargs):
    """Run training and evaluation."""
    trainer = unisal.train.Trainer(**kwargs)
    trainer.fit()
    for source in eval_sources:
        trainer.score_model(source=source)
        trainer.export_scalars()
        trainer.writer.close()


def load_trainer(train_id=None):
    """Instantiate Trainer class from saved kwargs."""
    if train_id is None:
        train_id = 'pretrained_unisal'
    print(f"Train ID: {train_id}")
    train_dir = Path(os.environ["TRAIN_DIR"])
    train_dir = train_dir / train_id
    return unisal.train.Trainer.init_from_cfg_dir(train_dir)


def predictions_from_folder(folder_path, is_video, source=None, train_id=None, model_domain=None):
    """Generate predictions of files in a folder with a trained model."""
    folder_path = Path(folder_path).resolve()
    is_video = bool(is_video)

    trainer = load_trainer(train_id)

    # ONLY register hook for video datasets
    if is_video:
        # Ensure RNN exists and runs
        trainer.model.bypass_rnn = False
        trainer.model.rnn.register_forward_hook(hook_fn)

    # Clear previous video features
    global rnn_feats
    rnn_feats = []

    trainer.generate_predictions_from_path(
        folder_path, is_video, source=source, model_domain=model_domain)

    # Save captured latent vectors to disk
    if is_video and rnn_feats:
        # Stack along time axis if needed
        stacked_feats = torch.cat(rnn_feats, dim=1)  # [batch, time, channels, h, w]
        video_name = folder_path.name
        save_path = LATENT_SAVE_DIR / f"{video_name}_rnn.pt"
        torch.save(stacked_feats, save_path)
        print(f"Saved RNN features for video {video_name} to {save_path}")


def predict_examples(train_id=None):
    """Generate predictions for example datasets (videos only)."""
    for example_folder in (Path(__file__).resolve().parent / "examples").glob("*"):
        if not example_folder.is_dir():
            continue

        source = example_folder.name
        is_video = source not in ('SALICON', 'MIT1003')  # Only videos

        print(f"\nGenerating predictions for video folder: {str(source)}")

        if is_video:
            if not example_folder.is_dir():
                continue
            for video_folder in example_folder.glob('[!.]*'):   # ignore hidden files
                predictions_from_folder(
                    video_folder, is_video, train_id=train_id, source=source)

    print("All video RNN features saved.")


if __name__ == "__main__":
    fire.Fire()
