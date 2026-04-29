# FedMedSeg: Potential Panel Questions

This document contains a comprehensive list of questions that a project panel might ask. Use these to test your knowledge and prepare your explanations.

---

### 🟢 1. General & Motivational Questions
1. What is the core problem your project is trying to solve?
2. Why did you choose Pneumonia detection specifically?
3. What is the main difference between Centralized and Federated Learning?
4. How does your project address the data privacy laws like DPDPA or HIPAA?
5. Why is Semantic Segmentation better than simple Image Classification for this problem?
6. What is the "Infection Burden," and why is it important for clinicians?

---

### 🟢 2. Data & Preprocessing Questions
1. What is a DICOM file, and why can't we just use JPEGs?
2. How did you handle the class imbalance in the RSNA dataset (many healthy vs. few sick)?
3. Why did you resize the images to $224 \times 224$? What happens if we use $1024 \times 1024$?
4. What normalization parameters did you use, and why?
5. How did you generate the ground-truth masks from the bounding box coordinates?
6. What is the "Non-IID" data distribution, and how did you simulate it in your code?

---

### 🟢 3. Model Architecture & Deep Learning Questions
1. Why did you choose the U-Net architecture for segmentation?
2. What is the role of the "Encoder" and the "Decoder" in a U-Net?
3. Explain the purpose of "Skip Connections." What happens if we remove them?
4. Why use MobileNetV2 as a backbone instead of ResNet or VGG?
5. What are "Depthwise Separable Convolutions," and how do they save computation?
6. What are "Inverted Residuals" and "Linear Bottlenecks"?
7. Explain the difference between Forward Propagation and Backward Propagation in your training loop.
8. What is a "Loss Function," and why did you use a combination of BCE and Dice?
9. What optimizer did you use, and what was your learning rate?

---

### 🟢 4. Federated Learning Questions
1. Explain the Flower (flwr) architecture. What is the role of the Server and the Client?
2. How are model weights aggregated in the central server?
3. What is "Client Drift," and when does it occur?
4. How does the FedProx algorithm differ from standard FedAvg?
5. What is the "Proximal Term" in FedProx, and what does the hyperparameter $\mu$ control?
6. How many rounds of federation did you perform?
7. What happens if one hospital node has a very slow internet connection or slow hardware?

---

### 🟢 5. Privacy & Security Questions
1. Does Federated Learning by itself guarantee 100% privacy?
2. What is a "Model Inversion Attack," and how can it reveal patient data?
3. What is Differential Privacy (DP), and how does it protect the model?
4. Explain the role of "Gradient Clipping" in DP-SGD.
5. Why do we add Gaussian Noise to the gradients before sending them to the server?
6. What does the "Epsilon" ($\epsilon$) value represent in your privacy budget?
7. What is the "Privacy-Utility Trade-off"? How much accuracy did you lose for privacy?

---

### 🟢 6. Results & Evaluation Questions
1. What is the Dice Coefficient, and how is it calculated?
2. Why is Pixel Accuracy sometimes a misleading metric for segmentation?
3. Explain your result: Why did FedProx perform better than isolated training?
4. How did you verify the model on "unseen" data?
5. Look at your training curves: Why do they oscillate more in Federated Learning compared to Centralized?
6. What was the "Inference Latency" of your system? Is it fast enough for real-time use?

---

### 🟢 7. Implementation & Deployment Questions
1. Why did you use Streamlit for the user interface?
2. Explain the "Alpha-Blending" logic for the red overlay.
3. How would you deploy this system across two different physical cities?
4. What are the hardware requirements to run your hospital client node?
5. What is the future scope of this project? (e.g., Asynchronous FL, Vision Transformers).
6. How would you handle a new hospital joining the federation after the model is already trained?
