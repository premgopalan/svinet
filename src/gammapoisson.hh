#ifndef GAMMAPOISSON_HH
#define GAMMAPOISSON_HH

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf.h>

#include "env.hh"
#include "matrix.hh"
#include "network.hh"

class GammaPoisson {
public:
  GammaPoisson(Env &env, Network &network);
  ~GammaPoisson(); 
  void infer();
  void em();
  void gen();
  
private:
  uint32_t duration() const;
  void set_heldout_sample(int s);
  void set_validation_sample(int s);
  void set_training_sample(int s);
  void init_gamma();
  void init_theta();
  void get_subsample(EdgeList &edges, uint32_t n);
  void get_random_edge(Edge &e) const;
  void get_random_pair(Edge &e) const;
  void process(EdgeList &sample);
  void update_phis(uint32_t p, uint32_t q, yval_t y);
  void set_gamma_exp0(const D3 &u, Matrix &v);
  void set_gamma_exp1(const D3 &u, Matrix &v);
  void add_phi_to_gammat(uint32_t p, uint32_t q);
  double approx_log_likelihood();
  double em_log_likelihood();
  void heldout_likelihood(double &a, double &b, double &c);
  void validation_likelihood(double &a, double &b, double &c);
  void training_likelihood(double &a, double &b, double &c);
  void compute_and_log_groups();
  void em_compute_and_log_groups();
  long factorial(long n);
  bool is_heldout(const Edge &e);
  bool is_validation(const Edge &e);
  double edge_likelihood(uint32_t p, uint32_t q, yval_t y) const;

  string edgelist_s(EdgeList &elist);
  
  Env &_env;
  Network &_network;
  
  uint32_t _n;
  uint32_t _k;
  uint32_t _t;
  uint32_t _s;
  uint32_t _iter;

  Array _alpha;
  Array _beta;
  Array _phi;
  D3 _gphi;

  D3 _gamma;
  D3 _gammat;
  Matrix _Elambda;
  Matrix _Eloglambda;

  double _ones_prob;
  double _zeros_prob;
  double _total_pairs;
  gsl_rng *_r;
  double _tau0;
  double _kappa;
  double _nodetau0;
  double _nodekappa;
  
  double _rhot;
  Array _noderhot;
  Array _nodec;

  time_t _start_time;
  time_t _last_iter;

  FILE *_lf;
  FILE *_hf;
  FILE *_vf;
  FILE *_hef;
  FILE *_vef;
  FILE *_tef;
  FILE *_tf;

  Matrix _theta;
  D3 _q;

  AdjMatrix _gen_y;
  Matrix _gen_lambda;
  Array _gen_alpha;
  Array _gen_beta;
  EdgeList _heldout_edges;
  EdgeList _validation_edges;
  EdgeList _training_edges;
  SampleMap _heldout_map;
  SampleMap _validation_map;
  SampleMap _training_map;
  mutable uint32_t _illegal;
  mutable uint32_t _illegal0;
  mutable uint32_t _good;
  mutable uint32_t _bad;
  mutable uint32_t _skip;
  double _max_t, _max_h, _max_v, _prev_h, _prev_w;
  uint32_t _nh;
};

inline
GammaPoisson::~GammaPoisson()
{
  fclose(_lf); 
  fclose(_hf); 
  fclose(_vf);
  fclose(_vef);
  fclose(_hef);
  fclose(_tef);
  fclose(_tf);
}

inline uint32_t
GammaPoisson::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline double
GammaPoisson::edge_likelihood(uint32_t p, uint32_t q, yval_t y) const
{
  const double **thetad = _theta.data();
  double s = .0;
  for (uint32_t k = 0; k < _k; ++k)
    s += thetad[p][k] * thetad[q][k];
  
  if (s < 1e-30)
    s = 1e-30;
  return y * log(s) - s;
}

// if (s < _env.epsilon && y > 0) {
//   const IDMap &m = _network.seq2id();
//   IDMap::const_iterator idt1 = m.find(p);
//   IDMap::const_iterator idt2 = m.find(q);
//   printf("%d %d\n", idt1->second,idt2->second);
//   _illegal++;
// }
//if (s == .0) {
//_illegal++;
//if (y == 0)
//_illegal0++;
//return .0;
//}  else
//_good++;

#endif
