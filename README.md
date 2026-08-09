# COMS W2132 Intermediate Computing in Python, Final Project 

## \<Protein-Protein Interaction Prediction Model>

### Author:
- [Sujeong Kim](https://github.com/coms2132-spring26/final-project-sk5755.git) <sk5755@columbia.edu>
 

## Project Category

#### Please select up to two items from the following list of categories that best describe your project:

* Data Analysis / Visualization
* Machine Learning

#### Please select which type of application best describes your project:
* Stand-alone Command Line Application 


## Project Abstract 

 
Understanding whether a certain protein interacts with another is important in many areas of biology, such as drug design and medicine. Researchers and scientists could use this model to check if a certain protein is reactive with another, and further look into the implications of the interaction. There are countless proteins in our bodies and accounting for every single interaction and effect can be difficult, so this tool will make it easier to track which proteins affect which. 
The user would input two proteins they want to test. Specifically, their amino acid sequence. Based on the two given amino acid sequence, the model will predict whether the two proteins will interact or not. When the proteins interact, they will bind to each other, leading to a conformational change or a signal cascade. 



## Scope / Challenges

* Define some specific deliverables of features you intent to build (In-Scope)

    : The user will input two amino acid sequences. Each amino acid is represented with a certain alphabet, and there are 20 amino acids in total.

    : The model will process the sequences and give an output interaction/no interaction


* Define some specific features you will specifically not build (Out-of-Scope) 

    : how or where the proteins will interact, and the result of the interaction. This requires a much higher level of understanding in molecular biology, rather than programming.

* Which part of the project do you expect to be easy and which parts do you think will be challenging or uncertain?

    : After the initial program is written, training the model with the data prepared will (hopefully) be easy. However, most data available on PPI(protein-protein interactions) are about proteins that already interact. So data of non-interacting proteins will have to be made using the data of the interacting proteins.

* How would you define "success" for the project even if you didn't complete all planned features?

    : Even if the prediction is inaccurate, I would consider the model to be successful if it read in the data and the user input correctly and quantified it so that the model could run. The accuracy of the model can be adjusted later. 

## Requirements / Dependencies 
* If there are any specific hardware, software, data sets, or online services / APIs you think you are going to use, please list them in the Requirements section. This includes any Python packages you will want to import. This section can be tentative for the initial proposal and the teaching staff can help identify resources.

    : I will require data of known protein-protein interactions, so I plan on using DIP(database of Interacting Proteins). The data is available in TSV format, so I will import it and use Pandas to read and analyze it. 

* How are you going to use AI in this project (for code generation, testing, or actually using an LLM for some part of your project)? 

    : Any part of the project that is out of the scope of this class. If a portion of the program requires a specific functionality that was not covered in class, using AI would help with the project. In addition, I plan on using an LLM to create test sets to measure the accuracy of the model. 

## Milestones 
1. Download the interacting protein data and clean/alter so that it is ready to process. Cleaning the data will include redundant data (same proteins in a different order or two of the same proteins)

2. Based on the amino acid sequence, represent each protein as numerical vectors, such as molecular weight, k-mer count, hydrophobicity, etc. This will also include processing the data further, such as altering the length of each protein so that each data point has the same length.

3. Store the data and its protein id in a hash map. 

4. Using scikit-learn, build a basic binary classifier. I plan on building multiple binary classifiers using molecular weight, k-mer count, hydrophobicity, and selecting the best model, or make a model that is a combination of multiple standards. 

5. Using PyTorch, build a deep learning model. 

6. Test the model.
