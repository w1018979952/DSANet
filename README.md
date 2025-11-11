# DSANet
Looking and Hearing into Details: Dual-enhanced Siamese Adversarial Network for Audio-Visual Matching


# requirments
python 3.8 \
librosa 0.7.2 \
numpy 1.19.0 \
torch 1.4.0 \
torchvision 0.5.0 

## Get Dataset and Paper
- Download the [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html), [VGGFace](https://www.dropbox.com/s/bqsimq20jcjz1z9/VGG_ALL_FRONTAL.zip?dl=0)
- Latest Paper List [Audio-visual matching](https://github.com/w1018979952/Audio-Visual-Matching)

#  VoxCeleb1
- wav audio data, 1,251 people in total, 39 GB after decompression.
- Baidu Cloud link: [VoxCeleb1](https://pan.baidu.com/s/1DtNDgQHfUmlbGsLsKA2nqA?pwd=wsie)
- VoxCeleb Document Classification：[VoxCeleb1 Document](https://pan.baidu.com/s/1UUJPGgyxquXgIcFtGnG_AQ?pwd=hm33)
- Decompression command:
- zip -s 0 split.zip --out unsplit.zip
- unzip unslit.zip

- Vox1 official website: [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)

# VoxCeleb2
- MP4 video data, files include audio, total of 5,994 people, 255 GB after decompression.
- Baidu Cloud link: [VoxCeleb2](https://pan.baidu.com/s/1o6AdzvkAsFY0fOByO6kMdQ?pwd=69q6)
- Decompression command:
- zip -s 0 vox2_mp4_dev.zip --out unsplit.zip
- unzip unslit.zip

- Vox2 official website: [VoxCeleb2](https://link.zhihu.com/?target=https%3A//www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
  
## Contact
If you have any questions, please feel free to contact with me at `Netizenwjx@foxmail.com`.

## Citation

```BibTeX
@article{wang2022looking,
  title={Looking and Hearing into Details: Dual-enhanced Siamese Adversarial Network for Audio-Visual Matching},
  author={Wang, Jiaxiang and Li, Chenglong and Zheng, Aihua and Tang, Jin and Luo, Bin},
  journal={IEEE Transactions on Multimedia},
  volume={25},
  pages = {7505-7516},
  year={2023}
}
