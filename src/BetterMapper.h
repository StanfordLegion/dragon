/*
 * Author: Joshua Payne
 * Adapted from default_mapper.h from the legion runtime.
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the  LANL Contributions to Legion (C15091) project.
 * See the LICENSE.txt file at the top-level directory of this distribution.
 */

#include <cstdlib>
#include <cassert>
#include <algorithm>

namespace LegionRuntime {
  namespace HighLevel {


    class BetterMapper : public ShimMapper {
    public:
    	BetterMapper(Machine machine, HighLevelRuntime *rt, Processor local);
    	BetterMapper(const BetterMapper &rhs);
      virtual ~BetterMapper(void);
    public:
      BetterMapper& operator=(const BetterMapper &rhs);
    public:
      virtual void select_task_options(Task *task);
      virtual void select_tasks_to_schedule(
                      const std::list<Task*> &ready_tasks);
      virtual void target_task_steal(
                            const std::set<Processor> &blacklist,
                            std::set<Processor> &targets);
      virtual void permit_task_steal(Processor thief, 
                                const std::vector<const Task*> &tasks,
                                std::set<const Task*> &to_steal);
      virtual void slice_domain(const Task *task, const Domain &domain,
                                std::vector<DomainSplit> &slices);
      virtual bool pre_map_task(Task *task);
      virtual void select_task_variant(Task *task);
      virtual bool map_task(Task *task);
      virtual bool map_copy(Copy *copy);
      virtual bool map_inline(Inline *inline_operation);
      virtual bool map_must_epoch(const std::vector<Task*> &tasks,
                            const std::vector<MappingConstraint> &constraints,
                            MappingTagID tag);
      virtual void notify_mapping_result(const Mappable *mappable);
      virtual void notify_mapping_failed(const Mappable *mappable);
      virtual bool rank_copy_targets(const Mappable *mappable,
                                     LogicalRegion rebuild_region,
                                     const std::set<Memory> &current_instances,
                                     bool complete,
                                     size_t max_blocking_factor,
                                     std::set<Memory> &to_reuse,
                                     std::vector<Memory> &to_create,
                                     bool &create_one,
                                     size_t &blocking_factor);
      virtual void rank_copy_sources(const Mappable *mappable,
                      const std::set<Memory> &current_instances,
                      Memory dst_mem, 
                      std::vector<Memory> &chosen_order);
      virtual bool speculate_on_predicate(const Mappable *mappable,
                                          bool &spec_value);
      virtual void configure_context(Task *task);
      virtual int get_tunable_value(const Task *task, 
                                    TunableID tid,
                                    MappingTagID tag);

      virtual void handle_mapper_task_result(MapperEvent event,
                                             const void *result,
                                             size_t result_size);
    public:
      // Helper methods for building other kinds of mappers, made static 
      // so they can be used in non-derived classes
      // Pick a random processor of a given kind
      static Processor select_random_processor(
                              const std::set<Processor> &options, 
                              Processor::Kind filter, Machine machine);
      // Break an IndexSpace of tasks into IndexSplits
      static void decompose_index_space(const Domain &domain,
                              const std::vector<Processor> &targets,
                              unsigned splitting_factor, 
                              std::vector<Mapper::DomainSplit> &slice);

      void set_proc(Task *task, Processor::Kind pkind);

    public:
      const Processor local_proc;
      const Processor::Kind local_kind;
      const Machine machine;
      // The maximum number of tasks a mapper will allow to be stolen at a time
      // Controlled by -dm:thefts
      unsigned max_steals_per_theft;
      // The maximum number of times that a single task is allowed to be stolen
      // Controlled by -dm:count
      unsigned max_steal_count;
      // The splitting factor for breaking index spaces across the machine
      // Mapper will try to break the space into split_factor * num_procs
      // difference pieces
      // Controlled by -dm:split
      unsigned splitting_factor;
      // Do a breadth-first traversal of the task tree, by default we do
      // a depth-first traversal to improve locality
      bool breadth_first_traversal;
      // Whether or not copies can be made to avoid Write-After-Read dependences
      // Controlled by -dm:war
      bool war_enabled;
      // Track whether stealing is enabled
      bool stealing_enabled;
      // The maximum number of tasks scheduled per step
      unsigned max_schedule_count;
      // Maximum number of failed mappings for a task before error
      unsigned max_failed_mappings;
      std::map<UniqueID,unsigned> failed_mappings;
      // Utilities for use within the default mapper 
      MappingUtilities::MachineQueryInterface machine_interface;
      MappingUtilities::MappingMemoizer memoizer;
      MappingUtilities::MappingProfiler profiler;
      std::set<Processor> all_procs;
      std::set<Processor> all_cpus;
      std::set<Processor> all_gpus;

      std::vector<Processor> all_cpus_v;
      std::vector<Processor> all_gpus_v;
      std::map<int,std::set<Processor>> node_procs;
      std::set<Memory> all_mems;
      std::set<Memory> system_mems;
      std::map<Processor,Memory> all_sysmem;

      Memory global_memory;
      std::map<Processor,double> sched_map;

      unsigned iproc;
      unsigned inode;

      unsigned i_cpu;
      unsigned i_gpu;

      bool map_to_gpus;



    };

  };
};

#endif // __DEFAULT_MAPPER_H__

// EOF

