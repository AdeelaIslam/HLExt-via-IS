# Line Extraction in Handwritten Documents via Instance Segmentation
Python Implementation

This repositery provides a Python implementation of extraction of text lines from handwritten document images.

### [Line Extraction in Handwritten Documents via Instance Segmentation](https://doi.org/10.1007/s10032-023-00438-7)
#### Adeela Islam<sup>1</sup> · Tayaba Anjum<sup>1</sup> · Nazar Khan<sup>1</sup>



<sup>1</sup> Department of Computer Science, University of the Punjab, Lahore, 54000, Punjab,
Pakistan <br>

## Usage
 
##### 1. Create a folder to save trained models
##### 2. Set path of this folder and data folder in code
##### 3. Training

```bash
$ python lineExtractionMain.py
```
##### 4. Set path of test data and model folder in code
##### 5. Testing

```bash
$ python LineExtractionTest.py
```

##### 6. [Pretrained models can be found at master branch of this repository](https://github.com/AdeelaIslam/HLExt-via-IS/tree/master)

## Results
Line extraction results along with sample annotations on 11 different datasets
![Figure 1](https://github.com/AdeelaIslam/HLExt-via-IS/blob/main/images/fig1.PNG)


# Citation
If this work is useful for your research, please cite our [Paper](https://doi.org/10.1007/s10032-023-00438-7):
```bash
@article{islam2023line,
  title={Line extraction in handwritten documents via instance segmentation},
  author={Islam, Adeela and Anjum, Tayaba and Khan, Nazar},
  journal={International Journal on Document Analysis and Recognition},
  volume={},
  number={},
  pages={},
  year={2023},
  publisher={Springer, NY},
  url={https://doi.org/10.1007/s10032-023-00438-7}
}
```
