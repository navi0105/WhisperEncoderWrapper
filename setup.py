import setuptools

setuptools.setup(
    name="WhisperEncoderWrapper",
    description="A simple wrapper of OpenAI Whisper's encoder model.",
    version="0.0.2",
    author="Ivan Leong",
    author_email="ivan000105@gmail.com",
    readme="README.md",
    packages=setuptools.find_packages(exclude=["tests"]),
    license="MIT",
    url="",
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=[
        'torch',
        'tqdm',
        'openai-whisper',
        'transformers'
    ],
    extras_require={"dev": ["pytest"]}
)
