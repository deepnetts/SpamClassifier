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
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.data.DataSet;
import visrec.ri.ml.classification.BinaryClassifierNetwork;

/**
 * Spam  Classification example.
 * This example shows how to create binary classifier for spam classification, using Feed Forward neural network.
 * Data is given as CSV file.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class SpamClassifier {

    public static void main(String[] args) throws DeepNettsException, IOException {

        int numInputs = 57;
        int numOutputs = 1;
        
        // load spam data  set from csv file
        DataSet dataSet = DataSets.readCsv("spam.csv", numInputs, numOutputs, true);             

        // split data set into train and test set
        DataSet[] trainTest = dataSet.split(0.6, 0.4);
        
        // normalize data
        MaxNormalizer norm = new MaxNormalizer(trainTest[0]);
        norm.normalize(trainTest[0]);
        norm.normalize(trainTest[1]);
        
        // create instance of feed forward neural network using its builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(25, ActivationType.TANH)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        // set training settings
        neuralNet.getTrainer().setMaxError(0.2f)
                              .setLearningRate(0.01f);
        
        // start training
        neuralNet.train(trainTest[0]);
        
        // test network /  evaluate classifier
        EvaluationMetrics em = Evaluators.evaluateClassifier(neuralNet, trainTest[1]);
        System.out.println(em);
        
        // create binary classifier using trained network
        BinaryClassifier<float[]> binClassifier = new BinaryClassifierNetwork(neuralNet);        
        float[] testEmail = getTestEmailFeatures();
        Float result = binClassifier.classify(testEmail);
        System.out.println("Spam probability: "+result);        
    }
    
    static float[] getTestEmailFeatures() {
        float[] emailFeatures = new float[57];
        emailFeatures[56] = 1;
        return emailFeatures;
    }
}