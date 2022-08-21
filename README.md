An pytorch implementation of autoencoder model for denosing hand-written number images.

## model structure

<img width="448" alt="Screen Shot 2022-08-21 at 2 57 07 PM" src="https://user-images.githubusercontent.com/75982405/185781543-c9ffb3ea-1768-4807-ab09-450191a04120.png">

- pre-trained weights is in `log/2022-08-21T12-31-49/2022-08-21T12-31-49_weights.pt`

## Usage
#### denoise one image
```shell
python test.py --image_path [ image path ] --model [ model weights path ]

optional arguments:
  -h, --help                show this help message and exit
  --input_image INPUT_IMAGE input image to be denoise
  --model MODEL             path of model weights (.pt)
  --output_dir OUTPUT_DIR   output dir for 2 output image (noisy_image, denoised_image)
```

#### train
```shell
python train.py 

optional arguments:
  -h, --help               show this help message and exit
  --epochs EPOCHS          num of epochs
  --batch_size BATCH_SIZE  total batch size
  --lr LR                  learning rate
  --s                      store result image of the 1st val image in every epoch
```
- will create a folder in `log/` named after current date and time.
- will record 
  1. training log (.txt)
  2. results of the first val image at every epochs (if --s is set)
  3. model weights (.pt) after all training is done
  
#### make GIF of intermediate results
```shell
python convertToGIF.py --image_dir [ image dir from training ] --gif_name [ output GIF name ]
```
e.g. <img src="https://user-images.githubusercontent.com/75982405/185782493-31da7cd5-83bb-4f7d-9619-0697eb8224a6.gif" width="100" height="100"> 


## Result
normal image  |  noisy image  |  denoised image

<img src="https://user-images.githubusercontent.com/75982405/185782699-930a7d0e-3112-453d-82ca-b1873681d6f4.png" width="100" height="100">.   <img src="https://user-images.githubusercontent.com/75982405/185782702-6759f4ce-a5f4-44cf-8d9e-3a1ed118a996.png" width="100" height="100">.   <img src="https://user-images.githubusercontent.com/75982405/185782706-d5d7af3c-5564-48c5-a29b-4038ba02fd95.png" width="100" height="100">
