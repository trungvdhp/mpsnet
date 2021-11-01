# MPSNet: A light-weight deep learning-based network specialized for mobile phones to extract palm vein features from the saturation channel

## Introduction
This repository gives the implementation of our proposed Palm ROI Extraction method based on Key Vectors and Palm Vein Features Extraction method based on Deep Learning (MPSNet).
## Databases
We use 4 public palm databases(Tongji, PolyU, CASIA, XJTU-UP) and two our self-collected palm databases (NTUST-HP and NTUST-IP) for training and testing MPSNet.

Instruction for downloading databases as follows:

1. <strong>Tongji: </strong>[https://cslinzhang.github.io/ContactlessPalm/](https://cslinzhang.github.io/ContactlessPalm/)

2. <strong>PolyU: </strong> Email to [Prof. Ajay Kumar](mailto:Ajay.Kumar@polyu.edu.hk?Subject=Request%20for%20HK%20Polytechnic%20University%20MultiSpectral%20Palmprint%20Database) to request for HK Polytechnic University MultiSpectral Palmprint Database.

3. <strong>CASIA: </strong>[http://www.cbsr.ia.ac.cn/english/MS_PalmprintDatabases.asp](http://www.cbsr.ia.ac.cn/english/MS_PalmprintDatabases.asp)

4. <strong>XJTU-UP: </strong> Email to [Prof. Dexing Zhong](mailto:bell@xjtu.edu.cn?Subject=Request%20for%20Xian%20Jiaotong%20University%20Unconstrained%20Palmprint%20(XJTU-UP)%20Database) to request for Xian Jiaotong University Unconstrained Palmprint (XJTU-UP) Database.

5. <strong>Our databases: </strong>
NTUST Contactless RGB Palm Database contains 2,180 RGB images from 109 people captured by HTC One A9 (named NTUST-HP) and 5,760 RGB images from 144 people by iPhone 8+ (named as NTUST-IP) in an unconstrained manner with the flashlight. The total number of individuals in our database is 253. The images in our database were collected from volunteers from China, Taiwan, Vietnam, Thailand, Indonesia, and Mongolia aged 18 to 63 years old with various backgrounds, rotations, scales, and different illuminations. NTUST-HP database is captured in one session, 10 samples indoor for each palm. NTUST-IP is captured in two sessions, 10 samples for each palm indoor and 10 samples for each palm outdoor. The time interval between two sessions is about one week.

    <strong>* Researchers can refer to 100 samples of NTUST-HP database (10 classes in ['Images/100_samples/ntust-hp/'](https://github.com/trungvdhp/mpsnet/blob/main/Images/ntust-hp) folder) and 100 samples of NTUST-IP database (5 classes in ['Images/100_samples/ntust-ip/'](https://github.com/trungvdhp/mpsnet/blob/main/Images/ntust-ip) folder). To request full of our databases, please follow the [ANNEX-A terms for use of database](https://github.com/trungvdhp/mpsnet/blob/main/ANNEX_A_terms_for_use_of_database.docx) and [Database release agreement](https://github.com/trungvdhp/mpsnet/blob/main/Database_release_agreement.docx) files.</strong>

    <strong>* Here are some samples from our databases:</strong>

    <table>
        <tr>
            <td><img src="https://github.com/trungvdhp/mpsnet/blob/main/Images/ntust_hp_s4.png" alt="1" width = auto height = auto></td>
            <td><img src="https://github.com/trungvdhp/mpsnet/blob/main/Images/ntust_ip_s4.png" alt="2" width = auto height = auto></td>
            <td><img src="https://github.com/trungvdhp/mpsnet/blob/main/Images/ntust_ip_s1.png" alt="3" width = auto height = auto></td>
        </tr>
        <tr>
            <td><img src="https://github.com/trungvdhp/mpsnet/blob/main/Images/ntust_ip_s2.png" alt="4" width = auto height = auto></td>
            <td><img src="https://github.com/trungvdhp/mpsnet/blob/main/Images/ntust_ip_s3.png" alt="5" width = auto height = auto></td>
            <td><img src="https://github.com/trungvdhp/mpsnet/blob/main/Images/ntust_ip_s5.png" alt="6" width = auto height = auto></td>
        </tr>
    </table>

## Training

1. <strong>Prepare databases: </strong>
In ['Keras_Code\data\\'](https://github.com/trungvdhp/mpsnet/blob/main/Keras_Code/data/) folder, put palm vein ROI images of each database into the corresponding folder in the following format: 'Database_name\Palm_ID\X.Y', where 'Palm_ID' is the unique identifier of palm, 'X' is image filename, 'Y' is image extension. For example: "NTUST-IP\0001\1_01.tiff", 'database_name' is 'NTUST-IP', 'Palm_ID' is '0001', filename is '1_01', extension is 'tiff'.

    The names of databases for training are listed in the  file ['Keras_Code\data\train_folders.txt'](https://github.com/trungvdhp/mpsnet/blob/main/Keras_Code/data/train_folders.txt).

    The names of databases with the corresponding number of sessions for testing are listed in the  file ['Keras_Code\data\test_folders.txt'](https://github.com/trungvdhp/mpsnet/blob/main/Keras_Code/data/test_folders.txt)

    For example, if you want to run 'Open-set' testing (the test set is not included in the train set) on 'NTUST-IP' database (two sessions), then in the file ['train_folders.txt'](https://github.com/trungvdhp/mpsnet/blob/main/Keras_Code/data/train_folders.txt) you input the following lines:
        
        tongji
        polyu
        casia
        xjtu-up
        ntust-hp
    
    and in the file ['test_folders.txt'](https://github.com/trungvdhp/mpsnet/blob/main/Keras_Code/data/test_folders.txt) you input the line:

        ntust-ip 2
    
    where '2' is number of sessions.

2. <strong>Change the settings </strong>
in the file ['Keras_Code\options.py'](https://github.com/trungvdhp/mpsnet/blob/main/Keras_Code/option.py) : the default settings are as used in our paper, you can run different training sessions or models by changing the default values of 'train_session' and 'model_name' options in this file also. You can set the default value of 'train_session' option to be 1 for 'Closed-set' testing (the test set is included in the train set) or 2 for 'Open-set' testing (the test set is not included in the train set). The default value of 'model_name' option is one of the following names:
    
        mobilenet_v1
        mobilenet_v2
        mobilenet_v3
        mobilefacenet
        mpsnet
3. <strong>Train and test model: </strong>
After preparing databases and settings, you can train and test model by running file ['Keras_Code\train.py'](https://github.com/trungvdhp/mpsnet/blob/main/Keras_Code/train.py) or ['Keras_Code\train.ipynb'](https://github.com/trungvdhp/mpsnet/blob/main/Keras_Code/train.ipynb) (Jupyter Notebook file). The results are stored in the folder 'Keras_Code\results\session_X\model_name\\', where 'X' is the training session and 'model_name' is the name of model.

## Results
    The results will be updated after our paper is published.