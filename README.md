<!-- PROJECT SHIELDS -->
<a name="readme-top"></a>
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- HEADER -->
<br />
<div align="center">
  <h1 align="center">Playing_Cards_Detector</h1>
  <p align="center">
    A Computer-Vision Project for detecting Playing Cards and Applying it to Tarneeb (with yolov5)
  </p>
</div>
<br />

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project
Project for building Playing Card Objects detection model, and applying it in the context of the game 'Tarneeb'

all image annotation and augmentation work in the project is done using the following tools:
* makesense
* python scripts using opencv 
* Roboflow

Model is trained and built with yolov5m pretrained weights on top of a collection of 50000+ images

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Instructions for setting up predictor and playing Tarneeb.

### Prerequisites

you must have:
* python 3.7 or above
* webcam in computer device

### Installation

1. with CMD, navigate to root directory of the repo
2. make a virtual environment
```sh
python -3 -m venv py-env
```
3. install requirements 
```sh
pip install -r requirements.txt
```
4. activate virtual environment
```sh
py-venv\Scripts\activate
```
5. download weights file for the playing card detector model <a href="https://drive.google.com/uc?export=download&id=1-CASlZnJ9E4eyXDamMLOCAdW29_0Voas">here</a>
6. store the file in runs/train
7. back in root directory, navigate to yolov5_share
```sh
cd yolov_share
```
8. to start detection with webcam, type the following command
```sh
python detect.py --weights runs/train/best.pt --source 0
```
9. Upon launch of webcam window, you can play around with the detector (start playing tarneeb upon pressing t)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Collecting Data
- [x] Building Detector Model
- [x] Test with Sample Images and Webcam
- [x] Implement Tarneeb into the Detector
- [ ] Building Another Model with Tensorflow API
- [ ] Stylizing Webcam Output for Tarneeb

See the [open issues](https://github.com/MODAJ18/Playing_Cards/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Mohammad Almasri - [@linkedin](https://www.linkedin.com/in/mohammad-almasri-964867197/) - modaj18@gmail.com
Omar Alsabi - [@linkedin](https://www.linkedin.com/in/omar-alsaabi-32675b193/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/MODAJ18/Playing_Cards.svg?style=for-the-badge
[contributors-url]: https://github.com/MODAJ18/Playing_Cards/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/MODAJ18/Playing_Cards.svg?style=for-the-badge
[forks-url]: https://github.com/MODAJ18/Playing_Cards/network/members
[stars-shield]: https://img.shields.io/github/stars/MODAJ18/Playing_Cards.svg?style=for-the-badge
[stars-url]: https://github.com/MODAJ18/Playing_Cards/stargazers
[issues-shield]: https://img.shields.io/github/issues/MODAJ18/Playing_Cards.svg?style=for-the-badge
[issues-url]: https://github.com/MODAJ18/Playing_Cards/issues
[license-shield]: https://img.shields.io/github/license/MODAJ18/Playing_Cards.svg?style=for-the-badge
[license-url]: https://github.com/MODAJ18/Playing_Cards/blob/master/License.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/mohammad-almasri-964867197/
