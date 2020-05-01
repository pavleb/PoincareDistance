package si.fis.unm.weka;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.imagefilter.BinaryPatternsPyramidFilter;

public class Genesis {

	public static void main(String[] args) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader("/Users/pavleb/prj/Podatki/VsiJPG/Zobrazi.arff")); // preberi
																														// iz
																														// datoteke
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		data.setClassIndex(data.numAttributes() - 1); // doda indeks zadnjemu atributu

		BinaryPatternsPyramidFilter bpp = new BinaryPatternsPyramidFilter(); // new instance of filter
		// BPP.setOptions(options); // set options// za cross validation

        bpp.setInputFormat(data);
        bpp.setImageDirectory("/Users/pavleb/prj/Podatki/VsiJPG/");
        Instances imgFilt = Filter.useFilter(data, bpp);
        
        Remove r = new Remove();
		r.setAttributeIndices("1"); // odstrani prvi atribut - ime fotografije (?)
        r.setInputFormat(imgFilt);
        Instances newData = Filter.useFilter(imgFilt,r);

       
		/** klasifikacija - uporaba klasifikatorja IBk */

		// for (int k = 0; k < 11; k++)
		IBk knn = new IBk(1); // (k=1) zaenkrat
		knn.setWindowSize(0);
		knn.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new PoincareDistance());
        knn.buildClassifier(newData);

		/**
		 * razdeli podatke na testno in učno množico - uporaba 10 cross fold validation
		 */

		Evaluation eval = new Evaluation(newData);
		eval.crossValidateModel(knn, newData, 10, new Random(1));

		/**
		 * izpiši rezultate - Stratified cross-validation (tu so formalni rezultati:
		 * število pravilno in napačno razvrščenih primerkov, kappa statistika,
		 * povprečna napaka, relativna napaka, število vseh primerkov) - Detailed
		 * Accuracy By Class (tu so natančnejši rezultati: Recall, Precision, MCC, ROC
		 * Area, PRC Area)
		 */

		System.out.println("** Rezultati  **");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toClassDetailsString());

	}

}
