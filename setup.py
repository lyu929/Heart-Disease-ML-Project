from setuptools import setup
from pathlib import Path


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="heart-disease-robustness-project",
    version="1.0.0",
    author="Group 10",
    author_email="your_email@example.com",  
    description="Robust heart disease prediction under dataset shift using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",


    py_modules=[
        "app",
        "advisor",
        "data_loader",
        "evaluate",
        "model",
        "preprocess",
        "visualize",
    ],


    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "joblib>=1.3.0",
        "xgboost>=2.0.0",
        "imbalanced-learn>=0.11.0",
        "torch>=2.0.0",
        "scipy>=1.10.0",
    ],


    python_requires=">=3.9",


    entry_points={
        "console_scripts": [
            "heart-disease-app=app:main",
        ],
    },


    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    include_package_data=True,
    zip_safe=False,
)