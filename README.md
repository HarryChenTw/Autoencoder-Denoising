An pytorch implementation of autoencoder model for denosing hand-written number images.
## Result
normal image  |  noisy image  |  denoised image

<img src="https://user-images.githubusercontent.com/75982405/185782699-930a7d0e-3112-453d-82ca-b1873681d6f4.png" width="100" height="100">   <img src="https://user-images.githubusercontent.com/75982405/185782702-6759f4ce-a5f4-44cf-8d9e-3a1ed118a996.png" width="100" height="100">   <img src="https://user-images.githubusercontent.com/75982405/187011111-0e1cca12-e992-47f1-a2cf-22285dc6d435.png" width="100" height="100">
 
during training 

<img src="https://user-images.githubusercontent.com/75982405/187011089-b7ee120b-b2c5-4fab-a9d9-370f633cf409.gif" width="100" height="100"> 

## model structure

<img width="450" alt="Screen Shot 2022-08-26 at 10 42 39 PM" src="https://user-images.githubusercontent.com/75982405/186931637-73a16ba1-e114-4ef9-968f-0980203b208d.png">

- pre-trained weights is in `log/2022-08-26T22-01-11/2022-08-26T22-01-11_weights.pt`

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
e.g. <img src="https://user-images.githubusercontent.com/75982405/187011089-b7ee120b-b2c5-4fab-a9d9-370f633cf409.gif" width="100" height="100"> 


