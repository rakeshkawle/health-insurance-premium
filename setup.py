from setuptools import find_packages,setup  ### find_packages search the sub packages in your modules
from typing import List

HYPHONE_E_DOT='-e .'
def get_requirements(file_path:str)->List:
    requirements=[],
    with open (file_path) as obj_file:
        requirements=obj_file.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPHONE_E_DOT in requirements:
            requirements.remove(HYPHONE_E_DOT)
        
        return requirements



# for versioning setup
setup(
    name='Health Insurance Premium Prediction',
    version='0.0.1',
    author='Rakesh Kawle',
    author_email='kawlerakesh@gmail.com',
    install_required=get_requirements('requirements.txt'),
    packages=find_packages()


)


