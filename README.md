<br/>
<p align="center">
  <a href="https://github.com//Data-Wrangling-Data-Normalization-And-Transformation">
    <img src="" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Mastering Data Wrangling: The Art of Data Normalization and Transformation</h3>

  <p align="center">
    Transform Data Chaos into Clarity: Elevate Your Analysis with Expert Data Wrangling and Normalization Techniques!
    <br/>
    <br/>
    <a href="https://github.com//Data-Wrangling-Data-Normalization-And-Transformation"><strong>Explore the docs »</strong></a>
    <br/>
    <br/>
    <a href="https://github.com//Data-Wrangling-Data-Normalization-And-Transformation">View Demo</a>
    .
    <a href="https://github.com//Data-Wrangling-Data-Normalization-And-Transformation/issues">Report Bug</a>
    .
    <a href="https://github.com//Data-Wrangling-Data-Normalization-And-Transformation/issues">Request Feature</a>
  </p>
</p>

![Downloads](https://img.shields.io/github/downloads//Data-Wrangling-Data-Normalization-And-Transformation/total) ![Contributors](https://img.shields.io/github/contributors//Data-Wrangling-Data-Normalization-And-Transformation?color=dark-green) ![Stargazers](https://img.shields.io/github/stars//Data-Wrangling-Data-Normalization-And-Transformation?style=social) ![Issues](https://img.shields.io/github/issues//Data-Wrangling-Data-Normalization-And-Transformation) ![License](https://img.shields.io/github/license//Data-Wrangling-Data-Normalization-And-Transformation) 

## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

### Problem Statement

#### Background:
The Adult dataset, sourced from the UCI Machine Learning Repository, is a commonly used dataset for income prediction tasks. It contains census data and is used to predict whether an individual earns more than $50K per year. The dataset includes various features such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, native country, and income. This dataset poses several challenges typical in real-world data including missing values, categorical data that needs encoding, and numerical data that requires normalization.

#### Problem:
Develop a Python script to preprocess the Adult dataset for a machine learning task. The preprocessing steps should include:

1. Loading and inspecting the dataset for a preliminary understanding.
2. Identifying and handling missing values.
3. Visualizing the distribution of numerical features.
4. Normalizing numerical features to ensure they are on a similar scale.
5. One-hot encoding the categorical features, avoiding the dummy variable trap.
6. Combining the processed features into a final dataset ready for machine learning model training.
7. Ensuring the target variable 'income' is properly included in the final dataset.
8. Splitting the dataset into training and testing sets.

### Solution

#### Implementation Steps:

1. **Load the Dataset**:
    - Use `pandas` to load the dataset from the provided URL.
    - Assign column names and handle missing values marked as ' ?'.

2. **Initial Data Overview**:
    - Print the first few rows using `data.head()` for an initial inspection.
    - Use `data.isna().sum()` to check for missing values.

3. **Data Visualization**:
    - Use `matplotlib` and `seaborn` to plot histograms of numerical features, aiding in understanding their distributions.

4. **Handle Missing Values**:
    - Apply `data.dropna(inplace=True)` to remove rows with missing values, ensuring data integrity.

5. **Normalize Numerical Features**:
    - Apply `MinMaxScaler` to scale numerical features. This step is critical for models sensitive to the scale of input data.
    - Visualize the scaled data again to confirm the effect of normalization.

6. **Encode Categorical Features**:
    - Identify categorical features and use `OneHotEncoder` for encoding, setting `drop='first'` to avoid the dummy variable trap.
    - Transform these features into a new DataFrame and rename columns based on encoder output.

7. **Combine Processed Features**:
    - Drop original categorical columns from the dataset and concatenate the encoded DataFrame.
    - Verify that 'income', the target variable, is still present.

8. **Dataset Splitting**:
    - Use `train_test_split` to divide the dataset into features (`X`) and target (`y`), and then into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`).

9. **Error Handling**:
    - Implement checks to ensure the presence of the 'income' column post-processing. If absent, raise a `ValueError`.

#### Result:
The script effectively preprocesses the Adult dataset, making it ready for machine learning models. It handles typical data preprocessing tasks like handling missing values, feature scaling, and encoding, ensuring that the dataset is cleaned and standardized for optimal model performance. The inclusion of visualizations aids in understanding the feature distributions both before and after normalization. By the end of this process, the dataset is well-prepared for any further analysis or model training tasks, addressing the key challenges presented by this dataset.

This Python script performs various data wrangling tasks, including loading, preprocessing, normalizing, encoding, and splitting a dataset, specifically the Adult dataset (also known as the "Census Income" dataset).

### Code Breakdown

1. **Import Libraries**: 
    - `pandas`: For data manipulation and analysis.
    - `matplotlib.pyplot` and `seaborn`: For data visualization.
    - `MinMaxScaler` and `OneHotEncoder` from `sklearn.preprocessing`: For normalization and encoding of features.
    - `train_test_split` from `sklearn.model_selection`: To split the dataset into training and testing sets.

2. **Load the Dataset**:
    - The dataset is loaded from a URL using `pandas.read_csv`. Column names are provided, and missing values are handled (`na_values=' ?'`).

3. **Initial Data Overview**:
    - Prints the first few rows of the dataset and the count of missing values in each column.

4. **Data Visualization**:
    - Histograms are generated for all numerical features using `matplotlib` and `seaborn` to understand their distribution.

5. **Handling Missing Values**:
    - Rows with missing values are dropped using `data.dropna()`.

6. **Normalize Numerical Features**:
    - `MinMaxScaler` is applied to scale numerical features to a range between 0 and 1. 
    - After normalization, histograms are generated again to visualize the effect of scaling.

7. **Encode Categorical Features**:
    - Categorical features (except 'income') are one-hot encoded to transform them into a format that can be easily used by machine learning algorithms.
    - `drop='first'` is used in `OneHotEncoder` to avoid the dummy variable trap (reducing multicollinearity).

8. **Combining Encoded and Numerical Features**:
    - The original categorical features are dropped, and the encoded features are concatenated with the remaining dataset.

9. **Dataset Splitting**:
    - The dataset is split into features (`X`) and target (`y`, i.e., 'income') and then into training (`X_train`, `y_train`) and testing (`X_test`, `y_test`) sets.

10. **Final Dataset Output**:
    - Prints the first few rows of the training dataset for final verification.

11. **Error Handling**:
    - The script includes checks to ensure that the 'income' column is present in the dataset at relevant stages. If not, it raises a ValueError.

### Key Functionalities

- **Data Cleaning and Preprocessing**: Missing values are identified and handled, which is a crucial step in data preprocessing.
- **Normalization**: Normalization is applied to numerical features to ensure they're on a similar scale, which is particularly important for algorithms sensitive to the scale of input data.
- **Encoding**: Categorical variables are one-hot encoded, making them suitable for use in machine learning models.
- **Data Splitting**: The processed dataset is split into training and testing sets, which is a standard practice in machine learning to evaluate model performance.

This code serves as a comprehensive example of handling a real-world dataset through essential steps of data preprocessing and is particularly useful for tasks involving machine learning model preparation.

## Built With

## 🛠️ Built With

This project is a comprehensive demonstration of data wrangling, normalization, and transformation, utilizing a range of powerful tools and libraries. The following technologies and libraries were instrumental in bringing this project to life:

### Core Technologies

- **Python**: The backbone of our project, Python offers versatility and a rich ecosystem of libraries, making it ideal for data manipulation and analysis tasks.

### Data Analysis and Wrangling Libraries

- **Pandas**: A cornerstone in data manipulation, Pandas provides extensive capabilities for data processing, cleaning, and analysis, allowing us to efficiently handle our dataset.
- **NumPy**: Essential for numerical operations, NumPy supports our data manipulation tasks with its powerful array-processing capabilities.

### Data Visualization

- **Matplotlib**: This library enables us to create static, animated, and interactive visualizations in Python, which was crucial for data exploration and presenting insights.
- **Seaborn**: Built on top of Matplotlib, Seaborn provides a high-level interface for drawing attractive and informative statistical graphics.

### Machine Learning and Preprocessing Libraries

- **Scikit-learn**: A vital tool in our project, Scikit-learn is used for data preprocessing, including feature scaling and encoding. Its utility in machine learning model development also enhances our project's scope.

### Version Control

- **Git**: For efficient tracking of changes and collaborative development.
- **GitHub**: Hosts our repository and facilitates version control, issue tracking, and collaboration with contributors.

### Development Tools

- **Jupyter Notebook**: Used for writing and testing code in an interactive environment, Jupyter Notebook was instrumental in prototyping and demonstrating the workflow.
- **Visual Studio Code**: An advanced IDE that supported our coding with powerful editing, debugging, and source code control features.

---

### 🌟 Acknowledgments

We express our gratitude to the open-source community and the maintainers of these incredible tools that made this project not just possible, but also a joy to work on. Their hard work and dedication continue to inspire and enable developers and data scientists around the world.

## Getting Started

## 🚀 Getting Started

This section guides you through setting up your local machine for running and contributing to this project. By following these instructions, you'll have a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:
- **Python**: The project is developed in Python. You need to have Python [installed](https://www.python.org/downloads/).
- **Pip**: Python's package installer. It usually comes with Python, but if not, follow [these instructions](https://pip.pypa.io/en/stable/installation/).

### Installation

1. **Clone the Repository**

   Start by cloning the repository to your local machine. Open your terminal and run:

   ```sh
   git clone https://your-repository-url.git
   cd your-repository-directory
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

   It's good practice to create a virtual environment for your project. This keeps your dependencies organized and separate from other projects.

   - Create a virtual environment:

     ```sh
     python -m venv venv
     ```

   - Activate the virtual environment:

     - On Windows:

       ```sh
       .\venv\Scripts\activate
       ```

     - On macOS and Linux:

       ```sh
       source venv/bin/activate
       ```

3. **Install Required Packages**

   With your virtual environment activated, install the required packages using pip:

   ```sh
   pip install -r requirements.txt
   ```

   This command reads the `requirements.txt` file in your project directory and installs all the necessary libraries.

### Running the Project

Once the installation is complete, you're ready to run the project. 

- If you're using Jupyter Notebook:
  
  ```sh
  jupyter notebook
  ```
  
  Navigate to the notebook file in the Jupyter UI and run the cells.

- For a Python script, run:

  ```sh
  python your-script-name.py
  ```

### Running Tests

Explain how to run the automated tests for this system if applicable.

### Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

### Additional Documentation

For more information on the project setup, features, and usage, refer to the `docs` directory in the repository.

---

We hope this guide helps you to set up and enjoy working on the project. Happy coding!

## Roadmap

See the [open issues](https://github.com//Data-Wrangling-Data-Normalization-And-Transformation/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com//Data-Wrangling-Data-Normalization-And-Transformation/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com//Data-Wrangling-Data-Normalization-And-Transformation/blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](https://github.com//Data-Wrangling-Data-Normalization-And-Transformation/blob/main/LICENSE.md) for more information.

## Authors

* **Robbie** - *PhD Computer Science Student* - [Robbie](https://github.com/TribeOfJudahLion/) - **

## Acknowledgements

* []()
* []()
* []()
