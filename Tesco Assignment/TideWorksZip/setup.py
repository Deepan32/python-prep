from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def getRequirements(path:str) -> List[str]:
    """
    Utility function to create the package requirement
    from the requirement.txt
    """
    requirments = []
    with open(path) as reqs:
        reqs.readlines()
        reqs = [req.replace('\n','') for req in reqs]

    if HYPEN_E_DOT in reqs:
        reqs.remove(HYPEN_E_DOT)
        
    return reqs 

setup(
    name="TideWorks",
    version="0.0.1",
    author="Vinoth Loganathan",
    author_email="VinothLoganathanOfficial@outlook.com",
    requires=getRequirements("requirements.txt")
)