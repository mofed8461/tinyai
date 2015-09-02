#ifndef __TINYNEAT_HPP__
#define __TINYNEAT_HPP__

/* custom defines:
 * INCLUDE_ENABLED_GENES_IF_POSSIBLE - if during experiment you found that too many genes are
 *                                     disabled, you can use this option.
 * ALLOWING_RECURRENCY_IN_NETWORK    - allowing recurrent links 
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>

namespace neat {

	typedef struct {	
		unsigned int innovation_num = -1;

		unsigned int from_node = -1;
		unsigned int to_node = -1;
		double weight = 0.0;

		bool enabled = true;
	} gene;


	class genome;

	class pool {
	public:

		struct {
			float connection_mutate_chance = 0.25;
			float perturb_chance = 0.90;
			float crossover_chance = 0.75;
			float link_mutation_chance = 2.0;
			float node_mutation_chance = 0.50;
			float bias_mutation_chance = 0.40;
			float step_size = 0.1;
			float disable_mutation_chance = 0.4;
			float enable_mutation_chance = 0.2;			
		} mutation_rates;
		
		unsigned int input_size;
		unsigned int output_size;
		
		// pool's local random number generator
		std::random_device rd;		
		std::mt19937 generator;

		pool(){
			// seed the mersenne twister with 
			// a random number from our computer
			generator.seed(rd());	

		}

		/* evolutionary methods */
		genome crossover(const genome& g1, const genome& g2);
		void mutate_weight(genome& g);
		void mutate_enable_disable(genome& g, bool enable);
		void mutate_link(genome& g, bool force_bias);

	};


	class genome {
	public:
		unsigned int fitness = 0;
		unsigned int adjusted_fitness = 0;
		unsigned int global_rank = 0;

		unsigned int max_neuron;
		
		std::vector<gene> genes;
		pool* global_pool;

		genome() = delete;
		genome(pool* glob_pool){			
			global_pool = glob_pool;
			max_neuron = global_pool->input_size + global_pool->output_size;
		}
		genome(const genome&) = default;
		
	};
	
	class species {
	public:
		unsigned int top_fitness = 0;
		unsigned int stale_fitness = 0;
		unsigned int average_fitness = 0;
		std::vector<genome> genomes;

		species(){}
	};



	/* now the evolutionary functions itself */


	genome pool::crossover(const genome& g1, const genome& g2){
		// Make sure g1 has the higher fitness, so we will include only disjoint/excess 
		// genes from the first genome.
		// If the fitness is equal then we will include from both genomes.
		bool include_from_both = (g2.fitness == g1.fitness);				
		if (g2.fitness > g1.fitness)
			return crossover(g2, g1);
		
		genome child(this);

		auto it1 = g1.genes.begin();
		auto it2 = g2.genes.begin();

		// coin flip random number distributor
		std::uniform_int_distribution<int> coin_flip(1, 2);	

		for (; it1 != g1.genes.end() && it2 != g2.genes.end(); it1++){			
			
			// if innovation marks match, do the crossover, else include from the first
			// genome because its fitness is not smaller than the second's 
			
			if ((*it2).innovation_num == (*it1).innovation_num){
				// do the coin flip
				int coin = coin_flip(this->generator);
				

			#ifdef INCLUDE_ENABLED_GENES_IF_POSSIBLE
				if (coin == 2 && (*it2).enabled)
					child.genes.push_back(*it2);
				else
					child.genes.push_back(*it1);
			#else
				if (coin == 2)
					child.genes.push_back(*it2);
				else
					child.genes.push_back(*it1);
			#endif

			} else 			

				// as said before, we include from the first otherwise
				child.genes.push_back(*it1);

			// move forward if not match and include genes from the second genome
			// if their fitness are equal
			while ((*it2).innovation_num < (*it1).innovation_num && it2 != g2.genes.end()){
				if (include_from_both)
					child.genes.push_back(*it2);
				it2++;
			}			
		}

		child.max_neuron = std::max(g1.max_neuron, g2.max_neuron);		
		return child;	
	}

	unsigned int random_neuron(std::vector<gene>& genes, nonInput){
		// first neurons are inputs (sensors) and outputs

	}

	/* mutations */

	void pool::mutate_weight(genome& g){		
		float step = this->mutation_rates.step_size;
		std::uniform_real_distribution<float> real_distributor(0.0, 1.0);

		for (size_t i=0; i<g.genes.size(); i++){
			if (real_distributor(this->generator) < this->mutation_rates.perturb_chance) 
				g.genes[i].weight += real_distributor(this->generator) * step * 2 - step;
			else
				g.genes[i].weight =  real_distributor(this->generator)*4 - 2; 
			
		}
	}

	void pool::mutate_enable_disable(genome& g, bool enable){
		std::vector<gene*> v;
		for (size_t i=0; i<g.genes.size(); i++)
			if (g.genes[i].enabled != enable)
				v.push_back(&(g.genes[i]));

		std::uniform_int_distribution<int> distributor(0, v.size());
		v[distributor(this->generator)]->enabled = enable;
	}

	void pool::mutate_link(genome& g, bool force_bias){
		

	}	

} // end of namespace neat

#endif