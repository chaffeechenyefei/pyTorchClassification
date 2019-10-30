Implentmented models:

	DictLayer: Dict Layer: A Structured Dictionary Layer[cvprw][2018]

	NetVLAD: NetVLAD-CNN architecture for weakly supervised place recognition[cvpr][2016]

Arcface:

	Wide&Deep[2016] @ models/location_recommendation.py

Requirements:

	conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

	pip install pretrainedmodels imgaug scikit-learn tqdm opencv-python pandas

	pip install --upgrade scikit-image

Projects:

1. location recommendation:  main_location_company.py

	It build the relationship between location(item) and company(user). But each user only buy one item.

2. classification: main_furniture.py

	It does classification for furniture.

3. visual localization:

	NetVLAD + D2Net for image retrieval and spatial verification.





