# Adversarial-Image-Sandbox

Using the cifar-10, dataset, I created a basic image classifier that can predict the class of a given test image. I then created a Fast Gradient Signed Method Attack which creates a pertubation for an image that results in an incorrect prediction from the model. To combat this attack, I created a vaccine by training the models with images that have the pertubations along with clean images. Testing it resulted in a higher accuracy from images with pertubations.
Using a stock airplane image as a test subject, an epsilon of 5 resulted in an inaccurate prediction. After the vaccine, an epsilon of 5 resulted in an accurate prediction.

<img width="416" height="456" alt="Airplaneinput" src="https://github.com/user-attachments/assets/627e8687-f91d-4a10-8d3b-0fd7f98d8c88" />
<img width="416" height="456" alt="Airplane epsilon 5" src="https://github.com/user-attachments/assets/aa49fd66-7ea2-4e6c-ad36-ac2a00d0a0c3" />
<img width="416" height="456" alt="airplane epsilon 5 vaccine" src="https://github.com/user-attachments/assets/f820db29-20c1-430e-80f7-b230f09b1e03" />
<img width="416" height="456" alt="Airplane epsilon 2" src="https://github.com/user-attachments/assets/e2839921-0559-434a-a48f-5c75d0b45bfa" />
