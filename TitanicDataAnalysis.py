{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1> TITANIC SURVIVOR ANALYSIS </h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1> STEP 1 :</h1> \n",
    "\n",
    "#### so first here we import the libraries pandas and numpy "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### so first here we import the libraries pandas and numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1> STEP 2 :</h1> \n",
    "\n",
    "#### now we are using the pandas pd.read_csv to read the dataset and pd.head() to view the first 5 rows."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df = pd.DataFrame(pd.read_csv('train.csv'))\n",
    "Titanic_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 12)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Now we will first deal with missing values or NaN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we see the maximum missing values are in the Cabin and followed by the Age and the Embarked column."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     NaN\n",
       "1     C85\n",
       "2     NaN\n",
       "3    C123\n",
       "4     NaN\n",
       "Name: Cabin, dtype: object"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df['Cabin'].head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### We will be droping only those columns whose missing values are greater than 35% of pd.shape value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Cabin    687\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "drop_col=Titanic_df.isnull().sum()[Titanic_df.isnull().sum()>(35/100*Titanic_df.shape[0])]\n",
    "drop_col"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Cabin'], dtype='object')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "drop_col.index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.drop(columns=['Cabin'],inplace=True)\n",
    "Titanic_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We are still having null values of Age and Embarked, so first we deal with Age by substituting the missing values with the mean values of age  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    0\n",
       "Survived       0\n",
       "Pclass         0\n",
       "Name           0\n",
       "Sex            0\n",
       "Age            0\n",
       "SibSp          0\n",
       "Parch          0\n",
       "Ticket         0\n",
       "Fare           0\n",
       "Embarked       2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.fillna(Titanic_df.mean(),inplace = True)\n",
    "Titanic_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     889\n",
       "unique      3\n",
       "top         S\n",
       "freq      644\n",
       "Name: Embarked, dtype: object"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df['Embarked'].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We see that embarked is having value of S in 644 cases thus let us replace missing values of embarked with S"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    0\n",
       "Survived       0\n",
       "Pclass         0\n",
       "Name           0\n",
       "Sex            0\n",
       "Age            0\n",
       "SibSp          0\n",
       "Parch          0\n",
       "Ticket         0\n",
       "Fare           0\n",
       "Embarked       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df['Embarked'].fillna('S',inplace = True)\n",
    "Titanic_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### So we can finally conclude that all the null values are removed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "Titanic_df['Sex'].replace({'female':1,'male':0},inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name  Sex   Age  SibSp  Parch  \\\n",
       "0                            Braund, Mr. Owen Harris    0  22.0      1      0   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    1  38.0      1      0   \n",
       "2                             Heikkinen, Miss. Laina    1  26.0      0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    1  35.0      1      0   \n",
       "4                           Allen, Mr. William Henry    0  35.0      0      0   \n",
       "\n",
       "             Ticket     Fare Embarked  \n",
       "0         A/5 21171   7.2500        S  \n",
       "1          PC 17599  71.2833        C  \n",
       "2  STON/O2. 3101282   7.9250        S  \n",
       "3            113803  53.1000        S  \n",
       "4            373450   8.0500        S  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### In the above section we changed the value of male and female to 0 & 1 respectively."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Family</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name  Sex   Age  \\\n",
       "0                            Braund, Mr. Owen Harris    0  22.0   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    1  38.0   \n",
       "2                             Heikkinen, Miss. Laina    1  26.0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    1  35.0   \n",
       "4                           Allen, Mr. William Henry    0  35.0   \n",
       "\n",
       "             Ticket     Fare Embarked  Family  \n",
       "0         A/5 21171   7.2500        S       1  \n",
       "1          PC 17599  71.2833        C       1  \n",
       "2  STON/O2. 3101282   7.9250        S       0  \n",
       "3            113803  53.1000        S       1  \n",
       "4            373450   8.0500        S       0  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df['Family'] = Titanic_df['SibSp']+Titanic_df['Parch']\n",
    "Titanic_df.drop(columns=['SibSp','Parch'],inplace=True)\n",
    "Titanic_df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Family</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>PassengerId</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.005007</td>\n",
       "      <td>-0.035144</td>\n",
       "      <td>-0.042939</td>\n",
       "      <td>0.033207</td>\n",
       "      <td>0.012658</td>\n",
       "      <td>-0.040143</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <td>-0.005007</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.338481</td>\n",
       "      <td>0.543351</td>\n",
       "      <td>-0.069809</td>\n",
       "      <td>0.257307</td>\n",
       "      <td>0.016639</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>-0.035144</td>\n",
       "      <td>-0.338481</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.131900</td>\n",
       "      <td>-0.331339</td>\n",
       "      <td>-0.549500</td>\n",
       "      <td>0.065997</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>-0.042939</td>\n",
       "      <td>0.543351</td>\n",
       "      <td>-0.131900</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.084153</td>\n",
       "      <td>0.182333</td>\n",
       "      <td>0.200988</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>0.033207</td>\n",
       "      <td>-0.069809</td>\n",
       "      <td>-0.331339</td>\n",
       "      <td>-0.084153</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.091566</td>\n",
       "      <td>-0.248512</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>0.012658</td>\n",
       "      <td>0.257307</td>\n",
       "      <td>-0.549500</td>\n",
       "      <td>0.182333</td>\n",
       "      <td>0.091566</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.217138</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Family</th>\n",
       "      <td>-0.040143</td>\n",
       "      <td>0.016639</td>\n",
       "      <td>0.065997</td>\n",
       "      <td>0.200988</td>\n",
       "      <td>-0.248512</td>\n",
       "      <td>0.217138</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             PassengerId  Survived    Pclass       Sex       Age      Fare  \\\n",
       "PassengerId     1.000000 -0.005007 -0.035144 -0.042939  0.033207  0.012658   \n",
       "Survived       -0.005007  1.000000 -0.338481  0.543351 -0.069809  0.257307   \n",
       "Pclass         -0.035144 -0.338481  1.000000 -0.131900 -0.331339 -0.549500   \n",
       "Sex            -0.042939  0.543351 -0.131900  1.000000 -0.084153  0.182333   \n",
       "Age             0.033207 -0.069809 -0.331339 -0.084153  1.000000  0.091566   \n",
       "Fare            0.012658  0.257307 -0.549500  0.182333  0.091566  1.000000   \n",
       "Family         -0.040143  0.016639  0.065997  0.200988 -0.248512  0.217138   \n",
       "\n",
       "               Family  \n",
       "PassengerId -0.040143  \n",
       "Survived     0.016639  \n",
       "Pclass       0.065997  \n",
       "Sex          0.200988  \n",
       "Age         -0.248512  \n",
       "Fare         0.217138  \n",
       "Family       1.000000  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Family</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name  Sex   Age  \\\n",
       "0                            Braund, Mr. Owen Harris    0  22.0   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    1  38.0   \n",
       "2                             Heikkinen, Miss. Laina    1  26.0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    1  35.0   \n",
       "4                           Allen, Mr. William Henry    0  35.0   \n",
       "\n",
       "             Ticket     Fare Embarked  Family  alone  \n",
       "0         A/5 21171   7.2500        S       1      0  \n",
       "1          PC 17599  71.2833        C       1      0  \n",
       "2  STON/O2. 3101282   7.9250        S       0      1  \n",
       "3            113803  53.1000        S       1      0  \n",
       "4            373450   8.0500        S       0      1  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df['alone'] = [ 0 if Titanic_df['Family'][i]>0 else 1 for i in Titanic_df.index]\n",
    "Titanic_df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we see how the SibSp and Parch affected the values of the passenger being alone or is having family and accordinly it affects the chances of survival."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "alone\n",
       "0    0.505650\n",
       "1    0.303538\n",
       "Name: Survived, dtype: float64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.groupby(['alone'])['Survived'].mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So we can say the person who was alone had a lower chance of being alive than the person with family."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sex\n",
       "0    0.188908\n",
       "1    0.742038\n",
       "Name: Survived, dtype: float64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.groupby(['Sex'])['Survived'].mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we see that male passengers had a lower rate of survival than the female passengers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass\n",
       "1    84.154687\n",
       "2    20.662183\n",
       "3    13.675550\n",
       "Name: Fare, dtype: float64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.groupby(['Pclass'])['Fare'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass\n",
       "1    0.629630\n",
       "2    0.472826\n",
       "3    0.242363\n",
       "Name: Survived, dtype: float64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.groupby(['Pclass'])['Survived'].mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can also say that people having 1st class ticket hold the maximum chances of survival followed by class 2 and class 3. It is also visible that persons with higher fare values holds the higher priority of getting servived."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Embarked\n",
       "C    0.553571\n",
       "Q    0.389610\n",
       "S    0.339009\n",
       "Name: Survived, dtype: float64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Titanic_df.groupby(['Embarked'])['Survived'].mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Passengers boarding at Cherbourg survived ib better portion than others."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CONCLUSION"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<li> The person who was alone had a lower chance of being alive than the person with family.\n",
    "<li> We see that male passengers had a lower rate of survival than the female passengers.\n",
    "<li> We can also say that people having 1st class ticket hold the maximum chances of survival followed by class 2 and class 3. It is also visible that persons with higher fare values holds the higher priority of getting servived. Thus we can say the hierarchy was prioratized.\n",
    "<li> Passengers boarding at Cherbourg survived ib better portion than others."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
