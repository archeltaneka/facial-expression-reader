# Video Face Recognition
This project will detect faces in a video file. Through the ability of deep learning, you can train the model with your own custom dataset to detect and recognize faces.

## 1. Prerequisites
- Numpy v.1.19
- PyTorch v.1.6.0
- CUDA 10.2 (optional for GPU training)
- ffpyplayer (optional for audio playback)
- Dataset (Twitter Media Downloader)

## 2. Usage
1. Run Twitter Media Downloader or obtain your own dataset
2. Open playground.ipynb
3. Change the hyperparameters if necessary, run
4. Repeat until you get the best result

### (Optional) Twitter Media Downloader
I used Twitter Media Downloader to download a large number of images. It's great if you use Chrome for your daily browser, because there is an extension that you can install. You can follow this [link](https://chrome.google.com/webstore/detail/twitter-media-downloader/cblpjenafgeohmnjknfhpdbdljfkndig?hl=en) to download and install it in your Chrome browser

<container>
    <img src='./data/twitter_media_downloader.png'>
</container>

Open [Twitter](twitter.com) then create an account or sign in if you have one. Find your favorite actors/actresses or singers to download their images. On the very top beside the username account, you will see something like this

<container>
    <img src='./data/rcpc_twitter.png'>
</container>

Click 'Media', and you will be shown a prompt to download the images. You can filter the start date, include gifs or videos, retweets, etc.

<container>
    <img src='./data/twitter_media_prompt.png'>
</container>

If you are done, click 'Start' to start downloading the images.

## 3. Example
At the end of the project, you will be able to see the final results similar to something like this
<container>
</container>