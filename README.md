# Image-Inpainting

This project focuses on image inpainting, a technique used to restore missing or damaged parts of an image using deep learning models. The goal is to generate visually plausible completions for the missing regions. This model uses a context encoder GAN trained on the celebsA dataset. It uses a context encoder GAN (Generative Adversarial Network). It is deployed using Streamlit on Streamlit Community Cloud. 

App Link: https://imageinpaint.streamlit.app/

## CelebsA Dataset
The CelebA (CelebFaces Attributes) dataset is a large-scale dataset containing over 200,000 celebrity face images with rich attribute annotations. It includes 40 facial attribute labels, such as age, gender, and expressions, making it widely used for tasks like facial recognition, image generation, and inpainting.

## Generative Adversarial Networks (GANs)
Generative Adversarial Networks (GANs) are a class of deep learning models used for generating realistic data by leveraging two neural networks: a generator and a discriminator. The generator creates synthetic data, while the discriminator evaluates its authenticity by distinguishing between real and generated samples. These networks compete in a minimax game, gradually improving the generator’s ability to produce realistic outputs. GANs are widely used in image generation, style transfer, and data augmentation, making them a powerful tool in tasks like image inpainting and deepfake creation.

## Context-Encoder GANs
Context Encoder GANs are a specialized type of Generative Adversarial Networks (GANs) designed for image inpainting. They use an encoder-decoder architecture, where the encoder extracts feature representations from a corrupted image, and the decoder reconstructs the missing regions. The GAN framework helps refine the output by training the generator to produce realistic completions while the discriminator ensures high-quality, context-aware results. This approach improves structural coherence and texture consistency, making it effective for restoring missing image parts in various applications.

## Working of the Model
The image inpainting model follows a deep learning-based approach to predict missing regions in an image. The process involves the following steps:

* **Input Preprocessing:** The input image is converted to a size of 128 * 128 pixels and a square mask of size 32 * 32 pixels is applied to it.
  
* **Feature Extraction (Encoder):** The incomplete image is passed through an encoder, which extracts high-level feature representations. Convolutional layers help capture spatial information and patterns from the surrounding regions.

* **Context-Aware Reconstruction (Generator):** The generator network, typically based on an encoder-decoder structure, predicts the missing content using the extracted features. If a GAN is used, the generator learns to create visually coherent patches that blend seamlessly with the existing image.

* **Adversarial Training (Discriminator):** In the case of a GAN-based approach, a discriminator network evaluates whether the generated patches are real or fake by comparing them to actual image distributions. The generator improves iteratively by trying to fool the discriminator.

* **Reconstruction Loss:** Measures pixel-wise similarity between generated and real images (performance of generator). 

* **Adversarial Loss:** Ensures that the inpainted regions appear natural by penalizing unrealistic patches (performance of discriminator).

* **Joint Loss Function:** Is a combination of Reconstruction Loss (L<sub>R</sub>) and Adversarial Loss (L<sub>A</sub>). In this case we use a formula that prioritises Recounstruction Loss over Adversarial Loss as we want to focus on the correct generation of features.
          <p align="center">L = 0.999*L<sub>R</sub> + 0.001*L<sub>A</sub> </p>

* **Training:** The model is trained for 25 epochs over the training dataset, to minimise the joint loss function as described above.

* **Validation:** After each epoch the model is validated against the validation training set to check its performance on new data.

* **Testing:** After the model is done training, it is tested against new images from the sample dataset.

* **Final Output:** The refined, inpainted image is reconstructed, ensuring continuity with the surrounding pixels.

