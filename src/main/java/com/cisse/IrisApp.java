package com.cisse;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


// Modele Multi-Layer-perceptron utilisé lorsque le nombre entrees sont pas important
//car ce reseaux de neurones est fully connected
public class IrisApp {

	public static void main(String[] args) throws Exception {
		
		 int batchSize=1; //lot d'exemple a utiliser lors de l'entrainement 
		 int outputSize=3;   // Taille de la sortie 
		 int classIndex=4;
	     double learningRate=0.001; // Vitesse d'apprentissage
	     int inputSize=4;   
	     int numHiddenNodes=10;
	    // MultiLayerNetwork model;
	     int numbEpochs=60; //nombre d'entrainement a effectuer
	     InMemoryStatsStorage inMemoryStatsStorage;
	  
	     
	   //Creation de la configuration de notre reseau de neurones 
	     System.out.println("--- CREATION DE NOTRE MODEL DE RESEAU DE NEURONES --- ");
	     MultiLayerConfiguration configurationDuModel = new NeuralNetConfiguration.Builder()
	    		  // Generation d'une valeur aleatoire pour les poids des neurones 
	    		  .seed(123)
	    		  //Algo de retropropagation du Gradient avec la vitesse pour mettre a jour les poids des connexions afin de minimiser l'erreur
	    		  .updater(new Adam(learningRate))
	    		  .list() // Cree un reseau de neurone par couche
	    		  // Couche intermediaire toujours utiliser DenseLayer
	    		  .layer(0 , new DenseLayer.Builder()
				    		  .nIn(inputSize).nOut(numHiddenNodes)
				    		  .activation(Activation.SIGMOID)
				    		  .build()
	    		        )
	    		   
	    		// Couche 2 == couche intermediaire
	    		  .layer(1, new OutputLayer.Builder()
				    		  .nIn(numHiddenNodes).nOut(outputSize)
				    		  .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR) // Fonction de calcul des erreur
				    		  .activation(Activation.SOFTMAX)
				     		  .build()
	    		         )
	    		  .build();
	     
	     // creation  et initialisation du model , model multiLayer Perceptron
	     MultiLayerNetwork model = new MultiLayerNetwork(configurationDuModel);
	     model.init();
	     
	     
	     
	     
//	     //Suivit (Demarrage du serveur de monitoring du processus d'apprentissage) de l'apprentissage avec uiserver qui donne une vue graphique sur le port 9000
	     System.out.println("----SUIVIT DE L'APPRENTISSAGE VIA DES GRAPHES AVEC ND4J-----");	     
	     
	     UIServer uiServer = UIServer.getInstance();
	     inMemoryStatsStorage = new InMemoryStatsStorage(); //loguer les informations durant l'apprentissage
	    
	     uiServer.attach(inMemoryStatsStorage); // lier uiserver et inMemoryStateStorage
	     model.setListeners(new StatsListener(inMemoryStatsStorage));
	     
	    
	     
	     
	     
	     // Entrainement du model
	     System.out.println("--------- ENTRAINEMENT  DU MODEL EN CHARGEANT DES DATASET --------------");
	     
	      //Charger les donnees dans le dataset
	     File fileTrain = new ClassPathResource("iris-train.csv").getFile(); // recuperer le fichier a entrainer
	     RecordReader recordReaderTrain = new CSVRecordReader(); //Lire dans un fichier csv
	     recordReaderTrain.initialize(new FileSplit(fileTrain)); //Ajouter le fichier a entrainer en les decoupant ligne par ligne
	     DataSetIterator datasetIteratorTrain = //Ajout du dataset
	    		  new RecordReaderDataSetIterator(recordReaderTrain,batchSize,classIndex,outputSize);
	     
	      //Entrainer le model pendant un nombre d'essais ( nombre de cycle d'optimisations)
	     
	     for (int i = 0; i < numbEpochs; i++) {
	    	 model.fit(datasetIteratorTrain);	
		}
	     
	     
	System.out.println("------------DEBUT EVALUATION DU MODEL --------------");
	     
	     File fileTest = new ClassPathResource("irisTest.csv").getFile();
	     RecordReader recordReaderTest = new CSVRecordReader();
	     recordReaderTest.initialize(new FileSplit(fileTest));
	     DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest,batchSize,classIndex,outputSize);
	      Evaluation evaluation = new Evaluation(); //POur evaluer le model
	      
	      while (dataSetIteratorTest.hasNext()) {
			DataSet dataSetTest = dataSetIteratorTest.next(); //Parcourir le dataSet
			INDArray features = dataSetTest.getFeatures();   //Recuperer les entrees du dataset
			INDArray TargetLabels   = dataSetTest.getLabels();
			INDArray predictedLabels = model.output(features); // prediction de la sortie en fonction des entrees
			evaluation.eval(predictedLabels, TargetLabels);
			
		}
	      
	      System.out.println(evaluation.stats());
	      
//	      	      
//	    //PREDICTION AVEC UN MODEL PAS ENCORE VU
//	      
//	      INDArray dataApredire = Nd4j.create(new double[][] {
//	    	                            {5.0,3.5,1.3,0.3},
//	    	                            {5.5,2.6,4.4,1.2},
//	    	                            {6.2,3.4,5.4,2.3}
//	    	                            
//	                           });
//	      
//	      INDArray output = model.output(dataApredire); //recuperer la prediction
//	      int [] classes = output.argMax(1).toIntVector();  //Recuperer l'index Max des tableaux qui sont les classes
//	      
//	      
//	      for (int i = 0; i < classes.length; i++) {
//	    	  System.out.println(output);
//	    	  System.out.println("CLASSES:" + labels[classes[i]]);
//		} 
	      
	
	 
	 	 
	      // Enregistrement du model
     System.out.println("--------- ENREGISTREMENT  DU MODEL POUR UNE UTILISATION ULTERIEUR------------");
     ModelSerializer .writeModel(model, "irisModel.zip", true); //enregistrer le model avec son algo d'apprentissage updater
	      
//	     
//	     while(datasetIteratorTrain.hasNext()) {
//	    	 DataSet dataSet = datasetIteratorTrain.next();
//	    	 System.out.println(dataSet.getFeatures());
//	    	 System.out.println(dataSet.getLabels());
//	     }


	}

}
