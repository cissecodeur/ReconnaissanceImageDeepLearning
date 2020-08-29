package com.cisse;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class IrisPrediction {
	public static void main(String[] args) throws IOException {
		
		String [] labels = {"Iris-setosa","Iris-versicolor","Iris-virginica"}	    ; //Noms des labels
	
		 // Utilisation d'un model prentrainé  pour la prediction en chargeant le cerveau (irisModel)
	    MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("irisModel.zip"));
	
 	      INDArray dataApredire = Nd4j.create(new double[][] {
          {5.0,3.5,1.3,0.3},
          {5.5,2.6,4.4,1.2}, 
	      {6.2,3.4,5.4,2.3}
        
   });
 	      
 	     
	      INDArray output = model.output(dataApredire); //recuperer la prediction
	      int [] classes = output.argMax(1).toIntVector();  //Recuperer l'index Max des tableaux qui sont les classes
	      
	      
	      for (int i = 0; i < classes.length; i++) {
	    	  System.out.println(output);
	    	  System.out.println("CLASSES:" + labels[classes[i]]);
 		} 

	}
	


}
