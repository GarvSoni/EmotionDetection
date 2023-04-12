## Real-Time Human Emotion Detection Project

### Create an Environment 
```
    conda create -p emotionDetection python==3.8    
```
### Activate the Environment 
```
    conda activate emotionDetection/
```
### To install requirements 
```
    pip install -r requirements.txt
```
### create a folder structure  like this
```
    main folder: - 
        config.py
        countImages.py
        main.py
        logs: - 
            Logs will come here.
        weights: - 
            Weights will come here and If you have weights and paste it here.
        dataset: - 
            Paste your dataset here.
```

### download weights from here
##### This is the link [link](https://drive.google.com/file/d/1uW0g6KdtyOqho2WjB5salujibr24hmgZ/view?usp=sharing)

### Download dataset from here
##### This is the dataset link I used [link](https://www.kaggle.com/datasets/msambare/fer2013)

### If you want to train more 
```
    python main.py --whattodo train_it
```
### If you want to test it
```
    python main.py --whattodo test_it
```