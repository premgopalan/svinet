#include "gammapoisson.hh"
#include "log.hh"
#include <sys/time.h>
#include <stdint.h>

GammaPoisson::GammaPoisson(Env &env, Network &network)
  : _env(env), _network(network),
    _n(env.n), _k(env.k), _t(env.t), _s(env.s),
    _iter(0), _alpha(_k), _beta(_k), _phi(_k),
    _gamma(_n,_k,_t), _gammat(_n,_k,_t),
    _Elambda(_n,_k), _Eloglambda(_n,_k),
    _gphi(_n,_n,_k),
    _ones_prob(.0), _zeros_prob(.0),
    _r(NULL),
    _tau0(env.tau0 + 1), _kappa(env.kappa),
    _nodetau0(env.nodetau0 + 1), _nodekappa(env.nodekappa),
    _rhot(.0), _noderhot(_n), _nodec(_n),
    _start_time(time(0)), 
    _last_iter(time(0)),
    _theta(_n,_k),
    _q(_n,_n,_k),
    _gen_y(_n,_n),
    _gen_lambda(_n,_k),
    _gen_alpha(_k), _gen_beta(_k),
    _illegal(0), _illegal0(0), _good(0), _bad(0), _skip(0),
    _max_t(-2147483647),
    _max_h(-2147483647),
    _max_v(-2147483647),
    _prev_h(-2147483647),
    _prev_w(-2147483647),
    _nh(0)
{
  // random number generation
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);

  _total_pairs = _n * (_n - 1);

  Env::plog("nodes",_n);
  Env::plog("groups",_k);
  Env::plog("minibatch",_s);
  Env::plog("total pairs", _total_pairs);

  for (uint32_t i = 0; i < _k; ++i) {
    _alpha[i] = env.default_shape * (1 + gsl_rng_uniform(_r));
    _beta[i] = env.default_rate * (1 + gsl_rng_uniform(_r));
  }

  Env::plog("alpha", _alpha);
  Env::plog("beta", _beta);

  _gphi.set_elements(1./_k);

  _ones_prob = double(_network.ones()) / _total_pairs;
  _zeros_prob = 1 - _ones_prob;

  Env::plog("ones_prob", _ones_prob);
  Env::plog("zeros_prob", _zeros_prob);

  init_gamma();
  set_gamma_exp0(_gamma, _Elambda);
  set_gamma_exp1(_gamma, _Eloglambda);

  info("gamma = %s", _gamma.s().c_str());
  info("Elambda = %s", _Elambda.s().c_str());
  info("Eloglambda = %s", _Eloglambda.s().c_str());

  Env::plog("network ones", _network.ones());
  Env::plog("network singles", _network.singles());

  _lf = fopen(Env::file_str("/logl.txt").c_str(), "w");
  if (!_lf)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }  
  _hf = fopen(Env::file_str("/heldout.txt").c_str(), "w");
  if (!_hf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }

  _vf = fopen(Env::file_str("/validation.txt").c_str(), "w");
  if (!_vf)  {
    printf("cannot open validation file:%s\n",  strerror(errno));
    exit(-1);
  }

  _tf = fopen(Env::file_str("/training.txt").c_str(), "w");
  if (!_vf)  {
    printf("cannot open training file:%s\n",  strerror(errno));
    exit(-1);
  }

  //approx_log_likelihood();

  init_theta();

  info("theta = %s", _theta.s().c_str());

  _hef = fopen(Env::file_str("/heldout-edges.txt").c_str(), "w");
  if (!_hef)  {
    printf("cannot open heldout edges file:%s\n",  strerror(errno));
    exit(-1);
  }

  _vef = fopen(Env::file_str("/validation-edges.txt").c_str(), "w");
  if (!_vef)  {
    printf("cannot open validation edges file:%s\n",  strerror(errno));
    exit(-1);
  }

  _tef = fopen(Env::file_str("/training-edges.txt").c_str(), "w");
  if (!_vef)  {
    printf("cannot open training edges file:%s\n",  strerror(errno));
    exit(-1);
  }

  set_heldout_sample(_env.heldout_ratio * _network.ones());
  set_validation_sample(_env.heldout_ratio * _network.ones());
  set_training_sample(_env.heldout_ratio * _network.ones());

  fprintf(_hef, "%s\n", edgelist_s(_heldout_edges).c_str());
  fprintf(_vef, "%s\n", edgelist_s(_validation_edges).c_str());
  fprintf(_tef, "%s\n", edgelist_s(_training_edges).c_str());

  fclose(_hef); 
  fclose(_vef); 
  fclose(_tef); 

  double a, b, c;
  heldout_likelihood(a,b,c);
  validation_likelihood(a,b,c);
  training_likelihood(a, b, c);
  Env::plog("heldout ratio", _env.heldout_ratio);
  Env::plog("heldout edges (1s and 0s)", _heldout_map.size());
  //debug("heldout edges = %s\n", edgelist_s(_heldout_edges).c_str());
}

void
GammaPoisson::init_gamma()
{
  double ***d = _gamma.data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t j = 0; j < _k; ++j)  {
      //d[i][j][0] = gsl_ran_gamma(_r, 400, 1./100);
      //d[i][j][1] = gsl_ran_gamma(_r, 200, 1./100);
      d[i][j][0] = _env.default_shape * (1 + gsl_ran_gamma(_r, 100, 1./100));
      d[i][j][1] = _env.default_rate * (1 + gsl_ran_gamma(_r, 100, 1./100));
    }
  fprintf(stdout, "init_gamma done\n");
}

string
GammaPoisson::edgelist_s(EdgeList &elist)
{
  ostringstream sa;
  for (EdgeList::const_iterator i = elist.begin(); i != elist.end(); ++i) {
    const Edge &p = *i;
    sa << p.first << "\t" << p.second << "\n";
  }
  return sa.str();
}

void
GammaPoisson::init_theta()
{
  double **d = _theta.data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      d[i][k] = gsl_ran_gamma(_r, 0.01, 1);
}

void
GammaPoisson::set_heldout_sample(int s)
{
  if (_env.accuracy)
    return;
  int c0 = 0;
  int c1 = 0;
  int p = s / 2;
  while (c0 < p || c1 < p) {
    Edge e;
    get_random_pair(e);
    
    uint32_t a = e.first;
    uint32_t b = e.second;
    assert (a != b);

#ifndef SPARSE_NETWORK
    const yval_t ** const yd = _network.y().const_data();
    yval_t y = yd[a][b] & 0x01;
#else
    yval_t y = _network.y(a,b);
#endif

    if (y == 0 and c0 < p) {
      c0++;
      _heldout_edges.push_back(e);
      _heldout_map[e] = true;
    }
    if (y == 1 and c1 < p) {
      c1++;
      _heldout_edges.push_back(e);
      _heldout_map[e] = true;
    }
  }
  printf("c0 = %d, c1 = %d\n", c0, c1);
  fflush(stdout);
}

void
GammaPoisson::set_validation_sample(int s)
{
  if (_env.accuracy)
    return;

  int c0 = 0;
  int c1 = 0;
  int p = s / 2;
  while (c0 < p || c1 < p) {
    Edge e;
    get_random_pair(e);

    Network::order_edge(_env, e);
    if (is_heldout(e))
      continue;
    
    uint32_t a = e.first;
    uint32_t b = e.second;
    assert (a != b);

#ifndef SPARSE_NETWORK
    const yval_t ** const yd = _network.y().const_data();
    yval_t y = yd[a][b] & 0x01;
#else
    yval_t y = _network.y(a,b);
#endif

    if (y == 0 and c0 < p) {
      c0++;
      _validation_edges.push_back(e);
      _validation_map[e] = true;
    }
    if (y == 1 and c1 < p) {
      c1++;
      _validation_edges.push_back(e);
      _validation_map[e] = true;
    }
  }
  printf("c0 = %d, c1 = %d\n", c0, c1);
  fflush(stdout);
}

void
GammaPoisson::set_training_sample(int s)
{
  if (_env.accuracy)
    return;

  int c0 = 0;
  int c1 = 0;
  int p = s / 2;
  while (c0 < p || c1 < p) {
    Edge e;
    get_random_pair(e);

    Network::order_edge(_env, e);
    if (is_heldout(e) || is_validation(e))
      continue;
    
    uint32_t a = e.first;
    uint32_t b = e.second;
    assert (a != b);

#ifndef SPARSE_NETWORK
    const yval_t ** const yd = _network.y().const_data();
    yval_t y = yd[a][b] & 0x01;
#else
    yval_t y = _network.y(a,b);
#endif

    if (y == 0 and c0 < p) {
      c0++;
      _training_edges.push_back(e);
      _training_map[e] = true;
    }
    if (y == 1 and c1 < p) {
      c1++;
      _training_edges.push_back(e);
      _training_map[e] = true;
    }
  }
  printf("c0 = %d, c1 = %d\n", c0, c1);
  fflush(stdout);
}

void
GammaPoisson::heldout_likelihood(double &a, double &a0, double &a1)
{
  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  for (SampleMap::const_iterator i = _heldout_map.begin(); 
       i != _heldout_map.end(); ++i) {
    const Edge &e = i->first;
    uint32_t p = e.first;
    uint32_t q = e.second;
    assert (p != q);

#ifndef SPARSE_NETWORK
    const yval_t ** const yd = _network.y().const_data();
    yval_t y = yd[p][q] & 0x01;
    bool seen = yd[p][q] & 0x80;
#else
    yval_t y = _network.y(p,q);
    bool seen = false; // XXX
#endif

    //assert (not seen);
    double u = edge_likelihood(p,q,y);
    s += u;
    k += 1;
    if (y) {
      sones += u;
      kones++;
    } else {
      szeros += u;
      kzeros++;
    }
    //printf("edge likelihood for (%d,%d) is %f\n", p,q,u);
    //fflush(stdout);
  }
  fprintf(_hf, "%d\t%d\t%.5f\t%d\t%.5f\t%d\t%.5f\t%d\n", 
	  _iter, duration(), s / k, k, 
	  szeros / kzeros, kzeros, sones / kones, kones);
  fflush(_hf);

  a = s / k;
  a0 = szeros / kzeros;
  a1 = sones / kones;

  bool stop = false;
  int why = -1;
  if (_iter > 100) {
    if (a > _prev_h && _prev_h != 0 && fabs((a - _prev_h) / _prev_h) < 0.00001) {
      stop = true;
      why = 0;
    } else if (a < _prev_h)
      _nh++;
    else if (a > _prev_h)
      _nh = 0;

    if (a > _max_h) {
      double av0, av1, av2;
      validation_likelihood(av0, av1, av2);
      
      double at0, at1, at2;
      training_likelihood(at0, at1, at2);

      _max_h = a;
      _max_v = av0;
      _max_t = at0;
    }
    
    if (_nh > 10) {
      why = 1;
      stop = true;
    }
  }
  _prev_h = a;
  FILE *f = fopen(Env::file_str("/max.txt").c_str(), "w");
  fprintf(f, "%d\t%d\t%.5f\t%.5f\t%.5f\t%.5f\t%d\n", 
	  _iter, duration(), 
	  a, _max_t, _max_h, _max_v, why);
  fclose(f);
  if (stop)
    exit(0);
}


void
GammaPoisson::validation_likelihood(double &av, double &av0, double &av1)
{
  if (_env.accuracy)
    return;

  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  for (SampleMap::const_iterator i = _validation_map.begin(); 
       i != _validation_map.end(); ++i) {
    const Edge &e = i->first;
    uint32_t p = e.first;
    uint32_t q = e.second;
    assert (p != q);

#ifndef SPARSE_NETWORK
    const yval_t ** const yd = _network.y().const_data();
    yval_t y = yd[p][q] & 0x01;
    bool seen = yd[p][q] & 0x80;
#else
    yval_t y = _network.y(p,q);
    bool seen = false; // XXX
#endif

    //assert (not seen);
    double u = edge_likelihood(p,q,y);
    s += u;
    k += 1;
    if (y) {
      sones += u;
      kones++;
    } else {
      szeros += u;
      kzeros++;
    }
    //printf("edge likelihood for (%d,%d) is %f\n", p,q,u);
    //fflush(stdout);
  }
  fprintf(_vf, "%d\t%d\t%.5f\t%d\t%.5f\t%d\t%.5f\t%d\n", 
	  _iter, duration(), s / k, k, 
	  szeros / kzeros, kzeros, sones / kones, kones);
  fflush(_vf);

  av = s / k;
  av0 = szeros / kzeros;
  av1 = sones / kones;
}

void
GammaPoisson::training_likelihood(double &av, double &av0, double &av1)
{
  if (_env.accuracy)
    return;

  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  for (SampleMap::const_iterator i = _training_map.begin(); 
       i != _training_map.end(); ++i) {
    const Edge &e = i->first;
    uint32_t p = e.first;
    uint32_t q = e.second;
    assert (p != q);

#ifndef SPARSE_NETWORK
    const yval_t ** const yd = _network.y().const_data();
    yval_t y = yd[p][q] & 0x01;
    bool seen = yd[p][q] & 0x80;
#else
    yval_t y = _network.y(p,q);
    bool seen = false; // XXX
#endif

    double u = edge_likelihood(p,q,y);
    s += u;
    k += 1;
    if (y) {
      sones += u;
      kones++;
    } else {
      szeros += u;
      kzeros++;
    }
    //printf("edge likelihood for (%d,%d) is %f\n", p,q,u);
    //fflush(stdout);
  }
  fprintf(_tf, "%d\t%d\t%.5f\t%d\t%.5f\t%d\t%.5f\t%d\n", 
	  _iter, duration(), s / k, k, 
	  szeros / kzeros, kzeros, sones / kones, kones);
  fflush(_tf);

  av = s / k;
  av0 = szeros / kzeros;
  av1 = sones / kones;
}

bool
GammaPoisson::is_heldout(const Edge &e)
{
  assert (e.first != e.second);
  const SampleMap::const_iterator u = _heldout_map.find(e);
  if (u != _heldout_map.end()) // not heldout edge
    return true;
  return false;
}

bool
GammaPoisson::is_validation(const Edge &e)
{
  assert (e.first != e.second);
  const SampleMap::const_iterator u = _validation_map.find(e);
  if (u != _validation_map.end()) // not heldout edge
    return true;
  return false;
}

void
GammaPoisson::em()
{
  double ***qd = _q.data();
  double **thetad = _theta.data();
  do {
    for (uint32_t i = 0; i < _n; ++i)
      for (uint32_t j = 0; j < _n; ++j) {
	
#ifndef SPARSE_NETWORK
	yval_t **yd = _network.y().data();
	yval_t y = yd[i][j] & 0x01;
#else
	yval_t y = _network.y(i,j);
#endif

	Edge e(i,j);
	Network::order_edge(_env, e);
	
	if (i != j && (is_heldout(e) ||is_validation(e)))
	  continue;
	  
	if (y) {
	  double w = .0;
	  for (uint32_t k = 0; k < _k; ++k) {
	    qd[i][j][k] = thetad[i][k] * thetad[j][k];
	    w += qd[i][j][k];
	  }
	  for (uint32_t k = 0; k < _k; ++k)
	    qd[i][j][k] /= w;
	}
      }

    info ("q = %s", _q.s().c_str());
    
    vector<double> s(_k, .0);
    for (uint32_t i = 0; i < _n; ++i)
      for (uint32_t k = 0; k < _k; ++k) {
	double w = .0;
	for (uint32_t j = 0; j < _n; ++j) {
#ifndef SPARSE_NETWORK
	  yval_t **yd = _network.y().data();
	  yval_t y = yd[i][j] & 0x01;
#else
	  yval_t y = _network.y(i,j);
#endif

	  Edge e(i,j);
	  Network::order_edge(_env, e);
	  if (i != j && (is_heldout(e) || is_validation(e)))
	    continue;
	  
	  if (y)
	    w += y * qd[i][j][k];
	}
	s[k] += w;
	thetad[i][k] = w;
      }
    for (uint32_t i = 0; i < _n; ++i)
      for (uint32_t k = 0; k < _k; ++k)
	thetad[i][k] /= sqrt(s[k]);

    _iter++;
    fprintf(stdout, "-- iter: %d --\n", _iter);
    fflush(stdout);
    _illegal = 0;
    _illegal0 = 0;
    _good = 0;
    _bad = 0;
    _skip = 0;
    if (_iter % _env.reportfreq == 0) {
      em_log_likelihood();
      em_compute_and_log_groups();
      double a, b, c;
      heldout_likelihood(a, b, c);
      validation_likelihood(a, b, c);
      training_likelihood(a, b, c);
      printf("illegal = %d, illegal0 = %d, good = %d, bad = %d, skip = %d\n", 
	     _illegal, _illegal0, _good, _bad, _skip);
      fflush(stdout);
    }
  } while (1);
}

double
GammaPoisson::em_log_likelihood()
{
  double **thetad = _theta.data();
  double ***qd = _q.data();
  double w = .0;
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t j = 0; j < _n; ++j) {
      if (i > j)
	continue;

      Edge e(i,j);
      Network::order_edge(_env, e);
      if (i != j && (is_heldout(e) || is_validation(e)))
	continue;

      for (uint32_t k = 0; k < _k; ++k) {
      
#ifndef SPARSE_NETWORK
	yval_t **yd = _network.y().data();
	yval_t y = yd[i][j] & 0x01;
#else
	yval_t y = _network.y(i,j);
#endif
	
	double s = thetad[i][k] * thetad[j][k];
	if (s == .0 && y == 0) {
	  //s = _env.epsilon;
	  continue;
	}  else if (s == .0 && y == 1) {
	  _bad++;
	  continue;
	}
	
	if (qd[i][j][k] == .0) {
	  _bad++;
	  continue;
	}
	
	w += y * qd[i][j][k] * log(s / qd[i][j][k]) - s;
      }
    }
  fprintf(stdout, "em log likelihood = %.5f\n", w);
  fflush(stdout);
  fprintf(_lf, "%d\t%d\t%.5f\n", _iter, duration(), w); 
  fflush(_lf);

  if (_env.accuracy && _iter > 10) {
    if (w > _prev_w && _prev_w != 0 && fabs((w - _prev_w) / _prev_w) < 0.00001) {
      FILE *f = fopen(Env::file_str("/done.txt").c_str(), "w");
      fprintf(f, "%d\t%d\t%.5f\n", _iter, duration(), w); 
      fclose(f);
      if (_env.nmi) {
	char cmd[1024];
	sprintf(cmd, "/usr/local/bin/mutual %s %s >> %s", 
		Env::file_str("/ground_truth.txt").c_str(),
		Env::file_str("/communities.txt").c_str(), 
		Env::file_str("/done.txt").c_str());
	if (system(cmd) < 0)
	  lerr("error spawning cmd %s:%s", cmd, strerror(errno));
      }
      exit(0);
    }
  }
  _prev_w = w;
  return w;
}

void
GammaPoisson::infer()
{
  EdgeList sample;
  while (1) {
    sample.clear();
    get_subsample(sample, _s);
    _rhot = pow(_tau0 + _iter, -1 * _kappa);
    process(sample);
    double scale = _total_pairs;
    double ***gd = _gamma.data();
    double ***gdt = _gammat.data();
    for (uint32_t i = 0; i < _n; ++i) {
      _noderhot[i] = pow(_nodetau0 + _nodec[i], -1 * _nodekappa);
      for (uint32_t k = 0; k < _k; ++k)
	for (uint32_t t = 0; t < _t; ++t) {
	  gdt[i][k][t] = (t ? _beta[k] : _alpha[k]) +	\
	    (scale / _s) * gdt[i][k][t];
	  gd[i][k][t] = (1 - _noderhot[i]) * gd[i][k][t] + \
	    _noderhot[i] * gdt[i][k][t];
	}
      _nodec[i]++;
    }
    _iter++;
    _last_iter = time(0);
    if (_iter % _env.reportfreq == 0) {
      printf("sample size = %d\n", (int)sample.size());
      printf("iteration = %d took %d secs\n", _iter, duration());
      approx_log_likelihood();
      compute_and_log_groups();
    }
  }
}

void
GammaPoisson::get_subsample(EdgeList &edges, uint32_t n)
{
  uint32_t i = 0;
  do {
    Edge e;
    get_random_edge(e);
    int a = e.first;
    int b = e.second;
    if (a != b) {
      edges.push_back(e);
      i++;
    }
  } while (i < _n);
}

void
GammaPoisson::get_random_pair(Edge &e) const
{
  do {
    e.first = gsl_rng_uniform_int(_r, _n);
    e.second = gsl_rng_uniform_int(_r, _n);
    //assert(e.first == e.second || Network::check_edge_order(e));
  } while (e.first == e.second);
  Network::order_edge(_env, e);
}

void
GammaPoisson::get_random_edge(Edge &e) const
{
  int p = gsl_rng_uniform_int(_r, _network.ones());
  //const Edge *elist = _network.edges();
  //assert(elist);
  const EdgeList &elist = _network.edges();
  e = elist[p];

#ifndef SPARSE_NETWORK
	  yval_t **yd = _network.y().data();
	  yval_t y = yd[e.first][e.second] & 0x01;
#else
	  yval_t y = _network.y(e.first,e.second);
#endif

  //  printf("y = %d", y);
  assert (y >= 1);
}


void
GammaPoisson::process(EdgeList &sample)
{
  _gammat.zero();
  set_gamma_exp0(_gamma, _Elambda);
  set_gamma_exp1(_gamma, _Eloglambda);

  uint32_t p, q;
  for (EdgeList::const_iterator i = sample.begin(); i != sample.end(); ++i) {
    p = i->first;
    q = i->second;
    assert (p != q);

#ifndef SPARSE_NETWORK
	  yval_t **yd = _network.y().data();
	  yval_t y = yd[p][q] & 0x01;
#else
	  yval_t y = _network.y(p,q);
#endif

    update_phis(p,q,y);
    add_phi_to_gammat(p, q);
    add_phi_to_gammat(q, p);
    //yd[p][q] = y | 0x80; // seen
  }
}

void
GammaPoisson::update_phis(uint32_t p, uint32_t q, yval_t y)
{
  const double ** const eloglambdad = _Eloglambda.const_data();
  //const double ** const elambad = _Elambda.const_data();
  _phi.set_elements(1./_k);
  double v = .0;
  for (uint32_t k = 0; k < _k; ++k) 
    v += _phi[k] * (eloglambdad[p][k] + eloglambdad[q][k] - log(_phi[k]));
  //info("before phi = %s\n", _phi.s().c_str());
  //info("before v = %f\n",v);

  double w = .0;
  for (uint32_t k = 0; k < _k; ++k) {
    _phi[k] = exp(eloglambdad[p][k] + eloglambdad[q][k]);
    w += _phi[k];
  }
  for (uint32_t k = 0; k < _k; ++k) 
    _phi[k] /= w;
  //_phi.lognormalize();
  //info("(%d,%d) -> phi = %s", p, q, _phi.s().c_str());
  _gphi.copy_slice(p,q,_phi);

  // check
  v = .0;
  for (uint32_t k = 0; k < _k; ++k) 
    v += _phi[k] * (eloglambdad[p][k] + eloglambdad[q][k] - log(_phi[k]));
  //info("after phi = %s\n", _phi.s().c_str());
  //info("after v = %f\n",v);
}

void
GammaPoisson::set_gamma_exp0(const D3 &u, Matrix &v)
{
  const double *** const d = u.const_data();
  double **e = v.data();
  for (uint32_t i = 0; i < u.m(); ++i)
    for (uint32_t j = 0; j < u.n(); ++j) {
      assert (d[i][j][1]);
      e[i][j] = d[i][j][0] / d[i][j][1];
    }
}

void
GammaPoisson::set_gamma_exp1(const D3 &u, Matrix &v)
{
  const double *** const d = u.const_data();
  double **e = v.data();
  for (uint32_t i = 0; i < u.m(); ++i)
    for (uint32_t j = 0; j < u.n(); ++j) {
      assert (d[i][j][1]);
      e[i][j] = gsl_sf_psi(d[i][j][0]) - log(d[i][j][1]);
    }
}

void
GammaPoisson::add_phi_to_gammat(uint32_t p, uint32_t q)
{
  double ***gdt = _gammat.data();
  double ***gd = _gamma.data();
  for (uint32_t k = 0; k < _k; ++k) {
    assert(gd[q][k][1]);
    gdt[p][k][0] += _phi[k];
    gdt[p][k][1] += _phi[k] * gd[q][k][0] / gd[q][k][1];
  }
}

long 
GammaPoisson::factorial(long n) 
{ 
  long v =  n <= 1 ? 1 : (n * factorial(n-1));
  //printf("factorial of %ld = %ld\n", n, v);
} 

double
GammaPoisson::approx_log_likelihood()
{
  if (_env.accuracy)
    return 0;

  const double *** const gphid = _gphi.const_data();
  const double *** const gd = _gamma.const_data();
  const double ** const eloglambdad = _Eloglambda.const_data();
  const double ** const elambdad = _Elambda.const_data();
  
  set_gamma_exp0(_gamma, _Elambda);
  set_gamma_exp1(_gamma, _Eloglambda);

  double s = .0;
  for (uint32_t p = 0; p < _n; ++p)
    for (uint32_t q = 0; q < _n; ++q) {
      const double * const phi = gphid[p][q];
      // info("phi = ");
      // for (uint32_t k = 0; k < _k; ++k)
      // info("%.3f", phi[k]);

#ifndef SPARSE_NETWORK
	  yval_t **yd = _network.y().data();
	  yval_t y = yd[p][q] & 0x01;
#else
	  yval_t y = _network.y(p,q);
#endif
      
      double v = .0;
      for (uint32_t k = 0; k < _k; ++k) 
	v += phi[k] * (eloglambdad[p][k] + eloglambdad[q][k] - log(phi[k]));
      s += v * y;
      
      for (uint32_t k = 0; k < _k; ++k)
	s -= elambdad[p][k] * elambdad[q][k];
      if (y) { 
	long f = factorial((long)y);
	//printf("y = %d, f = %ld, log f = %f\n", y, f, log(f));
	s = s - log(f);
      }
      //printf("s = %f\n", s);
    }

  for (uint32_t n = 0; n < _n; ++n)  {
    for (uint32_t k = 0; k < _k; ++k) {
      s += _alpha[k] * log(_beta[k]) + (_alpha[k] - 1) * eloglambdad[n][k];
      s -= _beta[k] * elambdad[n][k] + gsl_sf_lngamma(_alpha[k]);
    }
    for (uint32_t k = 0; k < _k; ++k) {
      s += gd[n][k][0] * log(gd[n][k][1]) + \
	(gd[n][k][0] - 1) * eloglambdad[n][k];
      s -= gd[n][k][1] * elambdad[n][k] + gsl_sf_lngamma(gd[n][k][0]);
    }
  }
  printf("approx. log likelihood = %.3f\n", s);
  fprintf(_lf, "%d\t%d\t%.5f\n", _iter, duration(), s); 
  fflush(_lf);

  info("gamma = %s", _gamma.s().c_str());
  info("Elambda = %s", _Elambda.s().c_str());
  info("Eloglambda = %s", _Eloglambda.s().c_str());
  info("updated gphi = %s", _gphi.s().c_str());

  return s;
}

void
GammaPoisson::compute_and_log_groups()
{
  const double *** const gphid = _gphi.const_data();
  FILE *groupsf = fopen(Env::file_str("/groups.txt").c_str(), "w");
  for (uint32_t i = 0; i < _n; ++i) {
    fprintf(groupsf,"%d\t", i);
    const IDMap &m = _network.seq2id();
    IDMap::const_iterator idt = m.find(i);
    assert (idt != m.end());
    fprintf(groupsf,"%d\t", (*idt).second);
    for (uint32_t k = 0; k < _k; ++k) {
      double s = .0;
      for (uint32_t j = 0; j < _n; ++j) {


#ifndef SPARSE_NETWORK
	  yval_t **yd = _network.y().data();
	  yval_t y = yd[i][j] & 0x01;
#else
	  yval_t y = _network.y(i,j);
#endif

	if (y)
	  s += gphid[i][j][k];
      }
      if (k == _k - 1)
	fprintf(groupsf,"%.3f\n", s);
      else
	fprintf(groupsf,"%.3f\t", s);
    }
  }
  fclose(groupsf);

  FILE *gammaf = fopen(Env::file_str("/gamma.txt").c_str(), "w");  
  const double *** const gd = _gamma.const_data();
  for (uint32_t i = 0; i < _n; ++i) {
    fprintf(gammaf,"%d\t", i);
    const IDMap &m = _network.seq2id();
    IDMap::const_iterator idt = m.find(i);
    assert (idt != m.end());
    fprintf(gammaf,"%d\t", (*idt).second);
    for (uint32_t k = 0; k < _k; ++k) {
      if (k == _k - 1)
	fprintf(gammaf,"%.3f\n", gd[i][k][0]/gd[i][k][1]);
      else
	fprintf(gammaf,"%.3f\t", gd[i][k][0]/gd[i][k][1]);
    }
  }
  fclose(gammaf);

  Array groups(_n);
  FILE *nratesf = fopen(Env::file_str("/nrates.txt").c_str(), "w");  
  for (uint32_t i = 0; i < _n; ++i) {
    fprintf(nratesf,"%d\t", i);
    const IDMap &m = _network.seq2id();
    IDMap::const_iterator idt = m.find(i);
    assert (idt != m.end());
    fprintf(nratesf,"%d\t", (*idt).second);
    double w = .0;
    for (uint32_t k = 0; k < _k; ++k)
      w += gd[i][k][0]/gd[i][k][1];
    double max = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      double v = gd[i][k][0]/(w * gd[i][k][1]);
      if (v > max) {
	max = v;
	groups[i] = k;
      }
      if (k == _k - 1)
	fprintf(nratesf,"%.3f\n", v);
      else
	fprintf(nratesf,"%.3f\t", v);
    }
  }
  fclose(nratesf);
  
  FILE *summaryf = fopen(Env::file_str("/summary.txt").c_str(), "a");
  D1Array<int> s(_k);
  for (uint32_t i = 0; i < _n; ++i)
    s[groups[i]]++;
  for (uint32_t i = 0; i < _k; ++i)
    fprintf(summaryf, "%d\t", s[i]);  
  fprintf(summaryf,"\n");
  fclose(summaryf);
}


void
GammaPoisson::em_compute_and_log_groups()
{
  FILE *commf = fopen(Env::file_str("/communities.txt").c_str(), "w");
  const IDMap &seq2id = _network.seq2id();
  MapVec communities;
  const double *** const dq = _q.const_data();
  FILE *groupsf = fopen(Env::file_str("/groups.txt").c_str(), "w");
  FILE *summaryf = fopen(Env::file_str("/summary.txt").c_str(), "a");
  for (uint32_t i = 0; i < _n; ++i) {
    fprintf(groupsf,"%d\t", i);
    const IDMap &m = _network.seq2id();
    IDMap::const_iterator idt = m.find(i);
    assert (idt != m.end());
    fprintf(groupsf,"%d\t", (*idt).second);
    for (uint32_t k = 0; k < _k; ++k) {
      double s = .0;
      for (uint32_t j = 0; j < _n; ++j) {
	
#ifndef SPARSE_NETWORK
	yval_t **yd = _network.y().data();
	yval_t y = yd[i][j] & 0x01;
#else
	yval_t y = _network.y(i,j);
#endif
	if (y)
	  s += dq[i][j][k];
      }
      if  (s > 1.0)
	communities[k].push_back(i);
      
      if (k == _k - 1)
	fprintf(groupsf,"%.3f\n", s);
      else
	fprintf(groupsf,"%.3f\t", s);
    }
  }
  fclose(groupsf);

  for (std::map<uint32_t, vector<uint32_t> >::const_iterator i = communities.begin();
       i != communities.end(); ++i) {
    const vector<uint32_t> &u = i->second;
    map<uint32_t, bool> uniq;
    for (uint32_t p = 0; p < u.size(); ++p) {
      map<uint32_t, bool>::const_iterator ut = uniq.find(u[p]);
      if (ut == uniq.end()) {
	IDMap::const_iterator it = seq2id.find(u[p]);
	uint32_t id = 0;
	assert (it != seq2id.end());
	id = (*it).second;
	fprintf(commf, "%d ", id);
	uniq[u[p]] = true;
      }
    }
    fprintf(commf, "\n");
  }
  fclose(commf);

  if (_env.nmi) {
    char cmd[1024];
    sprintf(cmd, "/usr/local/bin/mutual %s %s >> %s", 
	    Env::file_str("/ground_truth.txt").c_str(),
	    Env::file_str("/communities.txt").c_str(), 
	    Env::file_str("/mutual.txt").c_str());
    if (system(cmd) < 0)
      lerr("error spawning cmd %s:%s", cmd, strerror(errno));
  }
}

void
GammaPoisson::gen()
{
  double **gen_lambdad = _gen_lambda.data();
  yval_t **gen_yd = _gen_y.data();

  for (uint32_t k = 0; k < _k; ++k) {
    _gen_alpha[k] = 1.0;
    _gen_beta[k] = 1.0;
  }
  //int p = gsl_ran_uniform_int(_r, _k);
  //_gen_beta[p] = 1.0;
  
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k) {
      gen_lambdad[i][k] = gsl_ran_gamma(_r, _gen_alpha[k], _gen_beta[k]);
      printf("lambda = %.2f\n", gen_lambdad[i][k]);
    }
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t j = 0; j < _n; ++j) {
      double s = .0;
      if (i != j) {
	for (uint32_t k = 0; k < _k; ++k)
	  s += gen_lambdad[i][k] * gen_lambdad[j][k];
	gen_yd[i][j] = (int)(gsl_ran_poisson(_r, s));
      } else
	gen_yd[i][j] = 0;
      printf("s = %.2f, y = %d\n", s, gen_yd[i][j]);
    }
  info("gen y = %s\n", _gen_y.s().c_str());
  FILE *f = fopen(Env::file_str("/network_gen.dat").c_str(), "w");
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t j = 0; j < _n; ++j) {
      yval_t y = gen_yd[i][j];
      //printf("y = %d\n", y);
      if (y > 0) {
	if (y > 16)
	  y = 16;
	fprintf(f, "%d\t%d\t%d\n", i, j, y);
      }
    }
  fclose(f);
}


