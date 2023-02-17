# REPORT TEAM ID:53

## Abstract:

The vital extraction challenge aims to extract a patient's vitals from the image of an ICU monitor. Monitoring vitals is critical to providing high-quality care to patients and is essential for ensuring the best possible patient outcomes. While current guidelines state that the nurse-to-patient ratio should be 1:6, various practical issues result in a much worse scenario. This is why it is important to find newer and more efficient solutions to help solve this problem.

While it would be preferable to skip the camera-based monitoring system and directly feed the vitals into a common server, we recognize the fact that this solution is not an efficient solution for ICUs around India that are already up and running. There is a need to augment the existing ICU environments in order to capture the necessary vitals from an “offline” patient monitor and feed it into an “online” server for monitoring.

Precisely, this is the task assigned to us. This challenge required the development of a pipeline to first detect and segment the monitor's screen from the image and then detect, segment, and understand the various vitals present on the screen.

## Pipeline

![Pipeline_fin drawio (1)](https://user-images.githubusercontent.com/122287288/217323177-4d3aa8bd-5954-4043-aba3-a0a33584d34a.png)


We propose to solve the task with a 3-step solution, namely, Preprocessing, Vital Detection, and OCR. Lastly, we also tackled the HR Graph digitization.

### Preprocessing 

First, there is the segmentation of the monitor screen and separating it from the surroundings. Secondly, we have to scale the segmented screen to a uniform size for later stages while preserving the original resolution as much as possible. We apply the perspective transformation on the segmented image to obtain the resultant image.

### Vital Detection 

This stage in the pipeline aims to detect the appropriate vitals in the cropped and transformed monitor screen. However, there was a lot of unlabelled data, and manually annotating all 9000 images was not an option. And it will never be a good solution in any real-world scenario. We identified two methods to deal with the vast amount of unlabelled data. 

One approach was to use a customized Semi-Supervised Learning Model, which would learn from the small amount of labelled data and use it to generate pseudo labels and then true labels for the unlabelled data. 

Another approach was to intelligently pick and manually annotate a small amount of unlabelled data and then add it to the training dataset in order to bring maximum diversity and representation to the training data without much effort. This approach was chosen because of its better results and lesser computational requirement.

### OCR 
Finally, we applied an OCR to extract the values of the vitals. This was not a particularly difficult task considering the standardized fonts used on patient monitors. However, there were a few misreads in the final result, such as “0” being read as “o” or “O”, “1” being read as “I”, etc. Due to some detection inaccuracies, we would also see brackets creep into the detection box. Due to segmentation faults, sometimes the numbers at the left corner of the screen, most usually the Systolic pressure in monitors which display it at the bottom left, lose the hundreds’ place digit. We fix such issues by hard-coding a few logic-based checks and corrections to get the most logical possibility for the correct value.

### HR Graph Digitization 

Initially, we converted the HR graph segmented image into binary. In order to selectively obtain only the graph, the longest connected pixels row-wise were saved, and the rest discarded. The 2-D binary image was then projected into a 1-D Time Series. The plot was further rescaled in the x and y variables.


## Models Used:

Segmentation: We use the YOLO v8-n model for segmentation of monitor screens from input image. It gave 0.995 mAP50 and 0.987 mAP50-95 with inference time of around 120 ms. 

Preprocessing: The segmented image is then approximated into a quadrilateral (four points). We apply perspective transformation which brings vitals to more readable form.

Vital Detection: We use YOLO v8-s model for vital extraction from preprocessed image. It gave 0.988 mAP50 and 0.849 mAP50-95 with inference time of around 200 ms.

OCR of vitals: We use Paddle OCR because of its accuracy and speed. 

### Segmentation model metrics:

| Model | Epochs | Box Precision | Box Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLOv8n | 30 | 1 | 1 | 0.995 | 0.987 |
| YOLOv8s | 85 | 1 | 1 | 0.995 | 0.987 |
| Mask RCNN | 250 |  |  | 0.99 | 0.918 |

| Model | Epochs | IOU Loss | Mean Accuracy | Mean IOU |
|---|---|---|---|---|
| Segformer | 35 | 0.011 | 0.992 | 0.985 |

### Detection model metrics:

| Model | Epochs | Box Precision | Box Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLOv8n | 25 | 0.986 | 0.993 | 0.99 | 0.831 |
| YOLOv8s | 100 | 0.983 | 0.989 | 0.988 | 0.849 |
| YOLOv8m | 100 | 0.981 | 0.988 | 0.989 | 0.841 |
| YOLOv7 (R) |  | 0.991 | 0.996 | 0.993 | 0.83 |
| YOLOv6 | 200 |  |  | 0.954 | 0.721 |
| DETR | 45 |  |  | 0.979 | 0.688 |
| RetinaNet | 20 |  |  | 0.595 | 0.286 |

### Optical Character Recognition

| Model | Accuracy | Dataset | Inference Time |
|---|---|---|---|
| PaddleOCR | 74.80 | ICDAR15 | 8.54ms (per image) |

## Some other novel techniques we have tried:

## Applying Image processing to extract monitor screen

We can use the fact that the monitor is often of low saturation and is low in red and green channels to apply the filter.

This follows the use of the brown colour filter followed by a grey colour filter which would encompass most things in the foreground (white monitor boundaries and gray-ish and brown-ish walls and other items)
The problem with this approach, however, is that this generates a bounding box that would contain a tilted monitor screen for certain images. The tilted perspective caused a loss of accuracy which was not a favorable trade-off.

![WhatsApp Image 2023-02-07 at 23 32 42](https://user-images.githubusercontent.com/122287288/217328716-84db5210-7af1-450a-be77-16020414ee41.jpeg)


## Applying object detection for corners on semantic segmentation

We exploit the fact that the monitors will always be a quadrilateral to train the model on two bounding boxes, one for each of the two opposite corner points, converting the problem from semantic segmentation to object detection problem. This results in the  increase in speed of our model, however, it fails in particular use cases where it is unable to detect one of the objects due to overlap problems in particular niche use-cases. Since accuracy is the priority we opt for segmentation for a minimal time trade-off.

![WhatsApp Image 2023-02-07 at 23 32 50](https://user-images.githubusercontent.com/122287288/217328932-df292937-9954-4359-93f0-1ee30de3e36d.jpeg)

## Additional Annotation

We added around 800 labelled images with the help of the SCAN algorithm. SCAN (Semantic Clustering by Adopting Nearest-neighbour) is an unsupervised learning technique for image clustering.

We first made an embedding of images by training an encoder. After that, we found the nearest neighbour of each image and trained a model to find the clusters in the embedding. Finally, we took some images from each of the clusters and added them to our training dataset for better training.


## Semi-supervised Learning:

Semi-supervised learning is an efficient way to label unlabeled data. We first train an object detection model using available labelled data in this approach. Then, we make a copy of the trained model. One model is defined as the teacher model and the other one as the student. We pick a batch of unlabeled data and let the teacher model make predictions on this data. Now, the confident predictions (confidence value above a certain threshold) among these are assigned as Pseudo Labels. The student model is trained using an augmented version of these confident images and pseudo labels set as the target labels. The student model is allowed to train for 10 epochs. After 10 epochs, the weights of the teacher model are updated by the student model by Exponential Moving Average (EMA), and the process repeats. This way, the available unlabeled data is used to improve the model, which was already trained on available labelled data.

![SSL_fin drawio (1)](https://user-images.githubusercontent.com/122287288/217329013-482772e5-eb84-4c75-bac3-116c6434e1e5.png)


