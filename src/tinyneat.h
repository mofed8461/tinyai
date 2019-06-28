#ifndef __TINYNEAT_HPP__
#define __TINYNEAT_HPP__

/* custom defines:
 * INCLUDE_ENABLED_GENES_IF_POSSIBLE  - if during experiment you found that too many genes are
 *                                      disabled, you can use this option.
 * ALLOW_RECURRENCY_IN_NETWORK	      - allowing recurrent links 
 *
 * GIVING_NAMES_FOR_SPECIES           - giving species unique names (need a dictionary with 
 *                                      names in a file "specie_names.dict"
 */


#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <map>
#include <algorithm>
#include <list>
#include <string>

namespace neat {

	typedef struct {		
		double connection_mutate_chance = 0.25;
		double perturb_chance = 0.90;
		double crossover_chance = 0.75;
		double link_mutation_chance = 2.0;
		double node_mutation_chance = 0.50;
		double bias_mutation_chance = 0.40;
		double step_size = 0.1;
		double disable_mutation_chance = 0.4;
		double enable_mutation_chance = 0.2;			

		void read(std::ifstream& o);
		void write(std::ofstream& o, std::string prefix);
	} mutation_rate_container;

	typedef struct {
		uint32_t population = 240;
		double delta_disjoint = 2.0;
		double delta_weights = 0.4;
		double delta_threshold = 1.3;
		uint32_t stale_species = 15;

		void read(std::ifstream& o);
		void write(std::ofstream& o, std::string prefix);
	} speciating_parameter_container;

	typedef struct {
		uint32_t input_size;
		uint32_t bias_size;
		uint32_t output_size;
		uint32_t functional_nodes;
		bool recurrent;
	} network_info_container;

	typedef struct {	
		uint32_t innovation_num = -1;
		uint32_t from_node = -1;
		uint32_t to_node = -1;
		double weight = 0.0;
		bool enabled = true;
	} gene;

	class genome {
	private:
		genome();

	public:
		uint32_t fitness = 0;
		uint32_t adjusted_fitness = 0;
		uint32_t global_rank = 0;
		uint32_t max_neuron;
		uint32_t can_be_recurrent = false;

		mutation_rate_container mutation_rates;
		network_info_container network_info;

		std::map<uint32_t, gene> genes;

        genome(network_info_container& info, mutation_rate_container& rates);
		
		genome(const genome&) = default;
	};


	/* a specie is group of genomes which differences is smaller than some threshold */
	typedef struct {
		uint32_t top_fitness = 0;
		uint32_t average_fitness = 0;
		uint32_t staleness = 0;

	#ifdef GIVING_NAMES_FOR_SPECIES
		std::string name;
	#endif
		std::vector<genome> genomes;
	} specie;	

	class innovation_container {
	private:
		uint32_t _number;
		std::map<std::pair<uint32_t, uint32_t>, uint32_t> track;
        void set_innovation_number(uint32_t num);
		friend class pool;
	public:
        innovation_container();
        void reset();
        uint32_t add_gene(gene& g);
        uint32_t number();
	};


	/* a small world, where individuals (genomes) are making babies and evolving,
	 * becoming better and better after each generation :)
	 */
	class pool {
	private:
		pool();

		/* important part, only accecible for friend */
		innovation_container innovation;

		/* innovation tracking in current generation, should be cleared after each generation */
		std::map<std::pair<uint32_t, uint32_t>, uint32_t> track;


		uint32_t generation_number = 1;
		
		/* evolutionary methods */
		genome crossover(const genome& g1, const genome& g2);
		void mutate_weight(genome& g);
		void mutate_enable_disable(genome& g, bool enable);
		void mutate_link(genome& g, bool force_bias);
		void mutate_node(genome& g);		
		void mutate(genome& g);		

		double disjoint(const genome& g1, const genome& g2);
		double weights(const genome& g1, const genome& g2);
		bool is_same_species(const genome& g1, const genome& g2);

		/* specie ranking */
		void rank_globally();
		void calculate_average_fitness(specie& s);
		uint32_t total_average_fitness();

		/* evolution */
		void cull_species(bool cut_to_one);
		genome breed_child(specie& s);
		void remove_stale_species();
		void remove_weak_species();
		void add_to_species(genome& child);


	public:
		/* pool parameters */
		uint32_t max_fitness = 0;

		/* mutation parameters */
		mutation_rate_container mutation_rates;

		/* species parameters */
		speciating_parameter_container speciating_parameters;

		/* neural network parameters */
		network_info_container network_info;		

		// pool's local random number generator
		std::random_device rd;		
		std::mt19937 generator;

		/* species */
		std::list<specie> species;

		// constructor
		pool(uint32_t input, uint32_t output, uint32_t bias = 1, bool rec = false);

		/* next generation */
		void new_generation();
        uint32_t generation();

		/* calculate fitness */
        std::vector<std::pair<specie*, genome*> > get_genomes();

		/* import and export */
		void import_fromfile(std::string filename);
		void export_tofile(std::string filename);		
	};
} // end of namespace neat

#endif
