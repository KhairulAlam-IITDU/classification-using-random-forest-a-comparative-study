package twrf;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ReadFile {

	public List<List<Double>> features = new ArrayList<>();
	public List<Double> labels = new ArrayList<>();
	public DataNode dtnd = new DataNode();

	public void fileReader() {

		InputStream is = null;
		InputStreamReader isr = null;
		BufferedReader br = null;

		try {
			is = new FileInputStream("glass.txt");

			isr = new InputStreamReader(is);

			br = new BufferedReader(isr);

			String s = new String();

			while ((s = br.readLine()) != null) {

				String[] separate = s.split(","); // ","

				double decisionAttribute = Double.parseDouble(separate[0]); // [0], [separate.length-1]

				this.labels.add(decisionAttribute);

				List<Double> temp = new ArrayList<>();

				for (int i = 1; i < separate.length; i++) {					// int i = 0; i < separate.length-1; i++

					double matrixElement = Double.parseDouble(separate[i]);
					temp.add(matrixElement);

				}

				this.features.add(temp);

			}

			ArrayList<Integer> indices = new ArrayList<Integer>();

			for (int i = 0; i < this.labels.size(); i++) {
				indices.add(i);
			}

			Collections.shuffle(indices);

			for (int i = 0; i < indices.size(); i++) {

				this.dtnd.features.add(this.features.get(indices.get(i)));
				this.dtnd.labels.add(this.labels.get(indices.get(i)));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public DataNode[] makeFolds(int folds) {

		DataNode[] foldData = new DataNode[folds];

		int count = 0;
		int foldSize = (int) Math.ceil((double)this.dtnd.features.size()/folds);

		foldData[count] = new DataNode();

		for(int i = 0; i < this.dtnd.features.size(); i++){

			foldData[count].features.add(this.dtnd.features.get(i));
			foldData[count].labels.add(this.dtnd.labels.get(i));

			if((i+1)%foldSize == 0){
				count++;
				if(count != folds){
					foldData[count] = new DataNode();
				}
				else{
					count--;
				}
			}
			
		}

		return foldData;
	}

}
