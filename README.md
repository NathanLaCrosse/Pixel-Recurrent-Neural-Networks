# Pixel-Recurrent-Neural-Networks
In this repository, we explore the use of pixel recurrent networks to solve the infill problem.

## Dataset:
The datasets used were pulled from Kaggle, in particular:
MNIST Dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
Google Cartoon Faces: https://www.kaggle.com/datasets/brendanartley/cartoon-faces-googles-cartoon-set

The MNIST Dataset was used through most of the project due to training faster in comparison to the cartoon dataset. This project started with classification and was built upon until we created the final infilling model. Around when the architecture was updated to support infilling was when the dataset was switched to the cartoon dataset. 

The images, once requested, are altered in preparation for training. The images are loaded in as BGR due to cv2 reading images this way, so the color channels are flipped on input to RGB. To add, the 500x500 images are centered cropped down to 300x300 to remove much of the white space. The image is then resized down to 36x36 and lastly a row and column with color values 256 which are the number designated for start of string. Lastly the image is reconstructed into the format (Color, Row, Column). This reconstruction is done due to pytorch expecting images to be read in this format.

## Network Architecture:
Throughout the project, a lot of different types of architecture were considered. The general theme was the use of a RowRNN style approach - however even this approach has a lot of different choices that can be made. Let's start with the basics: how does a RowRNN work?

To begin, we consider an image row by row. For each row, we feed it to a Gated Recurrent Unit (GRU) which processes each pixel intensity (which is embedded into a vector) sequentially. In each sequential step, a hidden state vector is computed that acts as the GRU's memory. As the GRU continues throughout the row, it reads left to right, updating its memory at each step. At the end of the entire calculation, each of these hidden vectors will be converted into pixel intensities, which will fill in areas marked as 257 (an infill token). 

However, in order to capture the two-dimensional context present in images, we must expand our viewpoint to include data from other rows. Now, there are a lot of ways to do this - you can combine inputs via addition, concatenation, or even consider expanding the GRU with new gates to support the incoming data. For this project, we experimented with both concatenation (present in legacy designs) and a linear transformation + addition. Addition won out due to training speed improvements and simplification. Below is a visualization of a how the RowRNN processes a given row, with a previous row's hidden vectors. Note that for the first row these are all zero.

<img width="412" height="230" alt="networkarchitecture" src="https://github.com/user-attachments/assets/fa538c02-69b0-452c-9163-38cecd7345dc" />

This allows us to capture important two-dimensional context and produce believable infillings. However, as described, this model can only work with grayscale images. This was then scaled up to support full color images. First of all, we begin with three separate GRUs, one for each of the color channels. Each GRU is independently given its color channel to work with and produces a grid of hidden vectors. 

Using this straight away presents us with a problem - the predictions for each of the color channels are independent of each other, meaning results had inconsistent color choices. To reconcile this, these predictions have to be made conditional on each other. In other words, we determine the green channel intensity based off of what was already predicted for red and determine the blue channel intensity based off of what was already predicted for red and green. Below is a diagram depicting how this works on a single pixel across the three color channels.

<img width="403" height="174" alt="conditionalcolors" src="https://github.com/user-attachments/assets/ea66befd-1359-4d96-83b2-80d36ae2c6bc" />

More concretely, the process is as follows: 
  - Use the hidden vector (memory from GRU) from the red channel to predict a red intensity (via a linear layer).
  - Embed the predicted red intensity as 'red_pred'.
  - Use the hidden vector from the green channel concatenated with 'red_pred' to predict a green intensity (via another linear layer).
  - Embed the predicted green intensity as 'green_pred'.
  - Finally, use the hidden vector from the blue channel, concatenated with both 'red_pred' and 'green_pred' to predict a blue intensity.

As a side note, the final model uses a multi-layered GRU. In the high-up level of this commentary, everything said beforehand works the same. The only thing that changes is there is an extra dimension that the GRU is operating across that is hidden since we only consider the hidden vector produced by the GRU. This allows the model to become more expressive, which improves the model's ability to successfully model the problem at hand.

## Results:
