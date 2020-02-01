/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */
package com.deepnetts.examples.spamclassifier;

import deepnetts.data.DataSets;
import deepnetts.data.norm.MaxNormalizer;
import deepnetts.eval.Evaluators;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import javax.visrec.ml.ClassificationException;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.data.DataSet;
import visrec.ri.ml.classification.FeedForwardNetBinaryClassifier;


/**
 * Spam  Classification example.
 * This example shows how to create basic machine learning model for spam classification using neural networks.
 * Based on the set of example emails  CSV file given as data set, the model will learn to predict whether some given email is spam or not.
 *
 * What needs to be done in order to apply this for custom problem.1, 2, 3 ...
 * 
 * Step by step tutorial is available at.
 * Additional concise explanations of the basic machine learning steps and neural network concepts
 * are given as links in code comments.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class SpamClassifier {

    public static void main(String[] args) throws DeepNettsException, IOException, ClassificationException {

        int numInputs = 57; // 57 input features which are used to determine if email is a spam (for more detail see)
        int numOutputs = 1; // probability that email is a spam
        
        // Load data set from csv file.  What is a CSV file and dataset? <BlogLink>
        DataSet exampleEmails = DataSets.readCsv("spam.csv", numInputs, numOutputs, true);             

        // Split data set into train and test set. For more about why are we doing this see <BlogLink>
        DataSet[] trainAndTestSet = exampleEmails.split(0.7, 0.3);
        DataSet trainingSet = trainAndTestSet[0];
        DataSet testSet = trainAndTestSet[1];
        
        // Normalize data. To learn why are we doing this see <BlogLink>
        MaxNormalizer norm = new MaxNormalizer(trainingSet);
        norm.normalize(trainingSet);
        norm.normalize(testSet);
        
        // Create instance of feed forward neural network using its builder.To better understand neural network components and settings see this article.
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(30, ActivationType.TANH)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        // CONFIGURE: set training settings
        neuralNet.getTrainer().setMaxError(0.2f)
                              .setLearningRate(0.01f);
        
        // TRAIN: Start training Read this blog to understand Training output - da li sve ovo da ubacim u jedan clanak u kome imam vise paragrafa ili vise kracih clanaka od po 30 sec - 1 min
        neuralNet.train(trainingSet);
        
        // TEST: test network /  evaluate classifier  Undesrtand classifier evaluation metrics <blog article>
        EvaluationMetrics em = Evaluators.evaluateClassifier(neuralNet, testSet);
        System.out.println(em);
                
        // USE: create binary classifier using trained network, and use it through VisRec API
        // array is data structure that lacks semantics, and that exactly what Java developers like
        BinaryClassifier<float[]> binClassifier = new FeedForwardNetBinaryClassifier(neuralNet);        
        
        // get email to check if it is a spam
        float[] testEmail = getTestEmailFeatures(); // get it from test set
        Float result = binClassifier.classify(testEmail);
        
        System.out.println("Spam probability: "+result);        
    }
    
    static float[] getTestEmailFeatures() {
        float[] emailFeatures = new float[57];
        emailFeatures[56] = 1; // todo set 
        return emailFeatures;
    }
}