import java.io.IOException;

import ca.pfv.spmf.algorithms.frequentpatterns.fpgrowth.*;
import ca.pfv.spmf.patterns.itemset_array_integers_with_count.Itemset;

public class Fimrun {

	public static void main(String args[]) throws IOException
	{
		double startTimestamp = System.currentTimeMillis();
		
		AlgoFPGrowth alg = new AlgoFPGrowth();
		alg.runAlgorithm( "C:\\Users\\venka\\IdeaProjects\\Adbassg2\\Input200_1.txt", "C:\\Users\\venka\\IdeaProjects\\Adbassg2\\output200APmine_1.txt", 0.025);
		System.out.println("The run time for the prog is:"+(System.currentTimeMillis() - startTimestamp) );
	}
}