package twrf;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Main {

	public static void main(String[] args) {

		Scanner scn = new Scanner(System.in);
		
		System.out.println("Choose algorithm to run:");
		System.out.println("0 for Random Forest by Breiman");
		System.out.println("1 for Weighted Random Forest by Winham");
		System.out.println("2 for Trees Weighted Random Forest by Hong Bo Li");
		
		/*
		 * choice for algorithm
		 * rf	-> 0
		 * wrf	-> 1
		 * twrf	-> 2*/
		
		int choice = scn.nextInt();
		
		System.out.println("Enter fold number for cross validation");
		
		int folds = scn.nextInt();
		
		if(folds <= 0){
			System.out.println("Folds cannot be less than 1.");
			System.exit(0);
		}
		
		System.out.println("Enter file path (absolute):");
		
		String empty = scn.nextLine();
		String filepath = scn.nextLine();
		
		ReadFile r = new ReadFile();
		r.fileReader(filepath);
		
		scn.close();

		DataNode[] data = new DataNode[folds];
		DataNode[] traineeData = new DataNode[folds];
		DataNode[] testData = new DataNode[folds];

		data = r.makeFolds(folds);
		
		//make k-folds trainee and test data sets
		for(int i = 0; i < folds; i++){

			traineeData[i] = new DataNode();
			testData[i] = new DataNode();

			for (int j = 0; j < folds; j++) {

				if(i == j){

					for(List<Double> dts : data[j].features){
						testData[i].features.add(dts);
					}

					for(Double dts : data[j].labels){
						testData[i].labels.add(dts);
					}
				}
				else{

					for(List<Double> dts : data[j].features){
						traineeData[i].features.add(dts);
					}

					for(Double dts : data[j].labels){
						traineeData[i].labels.add(dts);
					}
				}
			}
		}
		//end of making k-folds trainee and test data sets

		int totalTestData = 0;
		int correctAnswer = 0;
		
		Map<Double, Integer> right = new HashMap<>();
		Map<Double, Integer> wrong = new HashMap<>();

		for (int j = 0; j < folds; j++) {

			System.out.println("Forest " + (j+1) + " is building...");

			Forest forest = new Forest();

			forest.makeDecisionTree(traineeData[j], choice);

			for (int i = 0; i < testData[j].features.size(); i++) {

				totalTestData++;
				
				double key = testData[j].labels.get(i);

				double returnedKey = forest.traverseForest(testData[j].features.get(i));

				if (key == returnedKey) {
					//System.out.println("\nOriginal class: " + key + "\tPredicted class: " + returnedKey + "\tRight Prediction.");
					
					if(right.containsKey(key)){
						right.put(key, right.get(key)+1);
					}
					else{
						right.put(key, 1);
					}
					
					correctAnswer++;
				}
				else {
					//System.out.println("Original class: " + key + "\tPredicted class: " + returnedKey + "\tOOPS! Wrong Prediction.");
					
					if(wrong.containsKey(key)){
						wrong.put(key, wrong.get(key)+1);
					}
					else{
						wrong.put(key, 1);
					}
				}
			}
		}

		double accuracyRate = ((double)correctAnswer / ((double)totalTestData)) * 100.0;
		System.out.println("\nTotal test Data = " + totalTestData + ",\tCorrect Answer = " + correctAnswer);
		System.out.println("\nAccuracy Rate = " + accuracyRate + "%");
		
		System.out.println("Right decisions: ");
		for(double key : right.keySet()){
			System.out.println("Key: " + key + "\tCount: " + right.get(key));
		}
		System.out.println("Wrong decisions: ");
		for(double key : wrong.keySet()){
			System.out.println("Key: " + key + "\tCount: " + wrong.get(key));
		}
	}
}