from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = f.read().splitlines()

with open("README.md", "r") as fh:
    long_discription = fh.read()

description = "Timestamp binning package"
setup(
    name="binit",
    description=description,
    long_discription=long_discription,
    long_discription_content_type="text/markdown",
    version="0.0.3",
    url="https://github.com/Ruairi-osul/binit",
    author="Ruairi O'Sullivan",
    author_email="ruairi.osullivan.work@gmail.com",
    include_package_data=True,
    license="GNU GPLv3",
    keywords="data-analysis data-science array timestamps alignment",
    project_urls={"Source": "https://github.com/Ruairi-osul/binit"},
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=reqs,
)
