from setuptools import setup, find_packages


def readme():
    with open('README.md', 'r') as f:
        return f.read()


setup(
    name='dm-aioaiagent',
    version='0.0.0',
    author='dimka4621',
    author_email='mismartconfig@gmail.com',
    description='This is my custom aioaiagent client',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://pypi.org/project/dm-aioaiagent',
    packages=find_packages(),
    install_requires=[
        'dm-logger>=0.6.6, <0.7.0',
        'python-dotenv>=1.0.0',
        'pydantic>=2.9.2, <3.0.0',
        'langchain>=1.0.0, <2.0.0',
        'langchain-core>=1.0.0, <2.0.0',
        'langchain-openai>=1.0.0, <2.0.0',
        'langgraph>=1.0.0, <2.0.0',
        'langsmith>=0.4.0, <1.0.0',
        'grandalf>=0.8.0, <0.9.0',
    ],
    extras_require={
        'anthropic': ['langchain-anthropic>=1.0.0, <2.0.0'],
        'gemini': [
            'langchain-google-genai>=4.0.0, <5.0.0',
            'langchain-google-vertexai>=3.0.0, <4.0.0',
        ],
        'groq':      ['langchain-groq>=1.0.0, <2.0.0'],
        'mistral':   ['langchain-mistralai>=1.0.0, <2.0.0'],
        'deepseek':  ['langchain-deepseek>=1.0.0, <2.0.0'],
        'ollama':    ['langchain-ollama>=1.0.0, <2.0.0'],
        'all': [
            'langchain-anthropic>=1.0.0, <2.0.0',
            'langchain-google-genai>=4.0.0, <5.0.0',
            'langchain-google-vertexai>=3.0.0, <4.0.0',
            'langchain-groq>=1.0.0, <2.0.0',
            'langchain-mistralai>=1.0.0, <2.0.0',
            'langchain-deepseek>=1.0.0, <2.0.0',
            'langchain-ollama>=1.0.0, <2.0.0',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords='dm aioaiagent',
    project_urls={
        'GitHub': 'https://github.com/MykhLibs/dm-aioaiagent'
    },
    python_requires='>=3.9'
)
