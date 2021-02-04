import os
import yaml
from setuptools import find_packages, setup
from datetime import datetime


def get_model_name():
	if os.path.exists('model-config.yml'):
		with open('model-config.yml') as file:
			data = yaml.load(file)
		model_name = data.get('model-name', 'default-name')
	else:
		model_name = 'default-name'

	cur_date = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
	model_name = f"{model_name}-{cur_date}"

	return model_name


REQUIRED_PACKAGES = []
model_name = get_model_name()

setup(
    name="trainer",
    version=0.1,
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Generic example trainer package.",
)
