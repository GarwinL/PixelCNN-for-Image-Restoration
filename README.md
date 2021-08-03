# PixelCNN for Image Restoration
A pytorch implementation for the application of PixelCNN++ to Image Restoration (Denoising, Inpainting, SISR)

## Setup
To run the code you need the following:
- Python 3
- Pytorch, tensorboardX

## Traning the model
Use `PixelCNNpp/train.py` to train the model. Datasets are not provided and have to be prepared before (see `PixelCNNpp/config.py`):
```
python PixelCNNpp/train.py
```

## Image Restoration (IR)
A trained network is provided in `Net/` to execute image restoration. <br />
Straightforward IR can be execute with `optimizer.py` of the corresponding folder `Denoising/`, `Inpainting/` and `SISR/`:
```
python optimizer.py
```

Additional files are available to train hyperparameters, perform IR evaluations or do patch denoising.

## References
- PixelCNNpp implementation based on ["A pytorch Implementation of PixelCNN++"](https://github.com/pclucas14/pixel-cnn-pp)
- Dataloader package was kindly provided by [Tobias Plötz](https://www.visinf.tu-darmstadt.de/team_members/tploetz/tploetz.en.jsp) (we modified the original package)
