#include "fastamm-sparsek.hh"
#include "log.hh"
#include <sys/time.h>

int FastAMMSparseK::ea = 0;
int FastAMMSparseK::eb = 0;

D3 *FPhiComp2::_gphi1 = NULL;
D3 *FPhiComp2::_gphi2 = NULL;

FastAMMSparseK::FastAMMSparseK(Env &env, Network &network)
  :_env(env), _network(network),
   _n(env.n), _k(env.k),
   _t(env.t), _s(env.s),
   _alpha(_k),
   _family(0), _prev_mbsize0(_s), _prev_mbsize1(_s),
   _eta(_k,_t),
   _gamma(_n,_k), _lambda(_k,_t),
   //_gcounts(_n, THRESH), _gsum(_n), 
   _tau0(env.tau0 + 1), _kappa(env.kappa),
   _nodetau0(env.nodetau0 + 1), _nodekappa(env.nodekappa),
   _rhot(.0), _noderhot(_n), _nodec(_n),
   _nodeupdatec(_n),
   _nodeupdate_rn(_n),
   _start_time(time(0)),
   _Elogpi(_n,_k), _Elogbeta(_k,_t), _Epi(_n,_k),
   _gammat(_n,_k), _szgammat(_n,_k), _gcounts(_n,_k),
   _lambdat(_k,_t), _count(_n),
   _Elogf(_k),
   _pcomp(env, _iter, _n, _k, _t, 0, 0, 0,
	  _scounts, _Elogbeta, _Elogf),
   _delaylearn_reported(false),
   _max_t(-2147483647),
   _max_h(-2147483647),
   _max_v(-2147483647),
   _prev_h(-2147483647),
   _prev_w(-2147483647),
   _prev_t(-2147483647),
   _nh(0), _nt(0),
   _training_done(false),
   _start_node(0), 
   _neighbors(_env.reportfreq)
{
  FPhiComp2::static_initialize(_n, _k);
  FastQueue::static_initialize(_k, _env.alpha);

  fprintf(stdout, "+ mmsb initialization begin\n");
  fflush(stdout);
  ea = 0;
  eb = 0;

  if (env.undirected)
    _total_pairs = _n * (_n - 1) / 2;
  else
    _total_pairs = _n * (_n - 1);

  fprintf(stdout, "+ running inference on %d nodes\n", _n);
  Env::plog("inference n", _n);
  Env::plog("total pairs", _total_pairs);

  _alpha.set_elements(env.alpha);
  printf("alpha set to %s\n", _alpha.s().c_str());
  fflush(stdout);
  _ones_prob = double(_network.ones()) / _total_pairs;
  _zeros_prob = 1 - _ones_prob;

  info("ones_prob = %.2f", _ones_prob);
  Env::plog("ones_prob", _ones_prob);
  Env::plog("zeros_prob", _zeros_prob);

  if (env.eta_type == "default") {
    env.eta0 = _total_pairs * _ones_prob / _k;
    env.eta1 = (_total_pairs / (_k * _k)) - _env.eta0;
    if (env.eta1 < 0)
      env.eta1 = 1.0;
  } else if (env.eta_type == "dense") {
    env.eta0 = env.eta0_dense;
    env.eta1 = env.eta1_dense;
  } else if (env.eta_type == "sparse") {
    env.eta0 = env.eta0_sparse;
    env.eta1 = env.eta1_sparse;
  } else if (env.eta_type == "regular") {
    env.eta0 = env.eta0_regular;
    env.eta1 = env.eta1_regular;
  } else {
    fprintf(stdout, "unknown eta_type\n");
    fflush(stdout);
    assert(0);
  }

  double **d = _eta.data();
  for (uint32_t i = 0; i < _eta.m(); ++i) {
    d[i][0] = env.eta0;
    d[i][1] = env.eta1;
  }
  info("eta = %s", _eta.s().c_str());
  Env::plog("eta", _eta);

  // random number generation
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);

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
  if (!_tef)  {
    printf("cannot open training edges file:%s\n",  strerror(errno));
    exit(-1);
  }

  init_heldout();
  printf("+ done heldout\n");

  if (_env.model_load)
    assert(load_model() >= 0);
  else {
    init_gamma();
    create_sparse_rep();
    assert (init_lambda() >= 0);
  }

  debug("gamma = %s", _gamma.s().c_str());
  debug("lambda = %s", _lambda.s().c_str());

  printf("+ done initializing gamma, lambda\n");

  // initialize expectations
  // set_dir_exp(_gamma, _Elogpi);
  set_dir_exp(_scounts, _Elogpi);
  set_dir_exp(_lambda, _Elogbeta);

  printf("+ done Elogpi and Elogbeta\n");

  debug("Elogpi = %s", _Elogpi.s().c_str());
  debug("Elogbeta = %s", _Elogbeta.s().c_str());

  _statsf = fopen(Env::file_str("/stats.txt").c_str(), "w");
  if (!_statsf)  {
    printf("cannot open stats file:%s\n",  strerror(errno));
    exit(-1);
  }

  _tf = fopen(Env::file_str("/time.txt").c_str(), "w");
  if (!_tf)  {
    printf("cannot open time file:%s\n",  strerror(errno));
    exit(-1);
  }

  _cf = fopen(Env::file_str("/convergence.txt").c_str(), "w");
  if (!_cf)  {
    printf("cannot open convergence file:%s\n",  strerror(errno));
    exit(-1);
  }

  _cmapf = fopen(Env::file_str("/cmap.txt").c_str(), "w");
  if (!_cmapf)  {
    printf("cannot open cmap file:%s\n",  strerror(errno));
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
  _trf = fopen(Env::file_str("/training.txt").c_str(), "w");
  if (!_trf)  {
    printf("cannot open training file:%s\n",  strerror(errno));
    exit(-1);
  }

  Env::plog("network ones", _network.ones());
  Env::plog("network singles", _network.singles());

  _lf = fopen(Env::file_str("/logl.txt").c_str(), "w");
  if (!_lf)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  _mf = fopen(Env::file_str("/modularity.txt").c_str(), "w");
  if (!_mf)  {
    printf("cannot open modularity file:%s\n",  strerror(errno));
    exit(-1);
  }

  //double a, b, c;
  //heldout_likelihood(a, b, c);
  //validation_likelihood(a, b, c);
  //training_likelihood(a, b, c);
  //save_model();
  //compute_and_log_groups();
#ifdef GLOBALPHIS
  approx_log_likelihood();
#endif

  gettimeofday(&_last_iter, NULL);
  _start_time = time(0);
  printf("+ mmsb initialization end\n");
}

FastAMMSparseK::~FastAMMSparseK()
{
  fclose(_statsf);
  fclose(_hf);
  fclose(_vf);
  fclose(_tf);
  fclose(_lf);
  fclose(_mf);
  fclose(_trf);
  fclose(_cf);
  fclose(_cmapf);

#ifdef MRSTATS
  fclose(_mrstatsf);
#endif
}

void
FastAMMSparseK::init_heldout()
{
  int s = _env.heldout_ratio * _network.ones();
  set_heldout_sample(s);
  //set_validation_sample(s);
  //set_training_sample(2*(_network.ones() - s));
  Env::plog("heldout ratio", _env.heldout_ratio);
  Env::plog("heldout edges (1s and 0s)", _heldout_map.size());
  fprintf(_hef, "%s\n", edgelist_s(_heldout_edges).c_str());
  fprintf(_vef, "%s\n", edgelist_s(_validation_edges).c_str());
  fprintf(_tef, "%s\n", edgelist_s(_training_edges).c_str());
  fclose(_hef);
  fclose(_vef);
  fclose(_tef);
}

string
FastAMMSparseK::edgelist_s(EdgeList &elist)
{
  ostringstream sa;
  for (EdgeList::const_iterator i = elist.begin(); i != elist.end(); ++i) {
    const Edge &p = *i;
    sa << p.first << "\t" << p.second << "\n";
  }
  return sa.str();
}

void
FastAMMSparseK::set_heldout_sample(int s)
{
  if (_env.accuracy)
    return;
  int c0 = 0;
  int c1 = 0;
  int p = s / 2;
  bool st = _env.stratified;
  while (c0 < p || c1 < p) {
    Edge e;
    if (c0 == p) {
      _family = 1;
      _env.stratified = true;
      get_random_edge(e, false);
    } else {
      get_random_edge(e, true);
    }

    uint32_t a = e.first;
    uint32_t b = e.second;
    yval_t y = get_y(a,b);

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
  _env.stratified = st;
  _family = 0;
}

void
FastAMMSparseK::set_validation_sample(int s)
{
  if (_env.accuracy)
    return;

  int c0 = 0;
  int c1 = 0;
  int p = s / 2;
  bool st = _env.stratified;
  while (c0 < p || c1 < p) {
    Edge e;
    if (c0 == p) {
      _family = 1;
      _env.stratified = true;
      get_random_edge(e, false);
    } else {
      get_random_edge(e, false);
    }
    uint32_t a = e.first;
    uint32_t b = e.second;
    yval_t y = get_y(a,b);

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
  _env.stratified = st;
  _family = 0;
}


void
FastAMMSparseK::set_training_sample(int s)
{
  int c0 = 0;
  int c1 = 0;
  int p = s / 2;
  bool st = _env.stratified;
  while (c0 < p || c1 < p) {
    Edge e;
    if (c0 == p) {
      _family = 1;
      _env.stratified = true;
      get_random_edge(e, false);
    } else {
      get_random_edge(e, false);
    }
    uint32_t a = e.first;
    uint32_t b = e.second;
    yval_t y = get_y(a,b);

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
  _env.stratified = st;
  _family = 0;
}

void
FastAMMSparseK::init_gamma()
{
  double **d = _gamma.data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t j = 0; j < _k; ++j)
      d[i][j] = gsl_ran_gamma(_r, 100, 1./100);
}

int
FastAMMSparseK::init_lambda()
{
  if (_lambda.copy_from(_eta) < 0) {
    lerr("init lambda failed");
    return -1;
  }
  return 0;
}

void
FastAMMSparseK::update_gmap(const double **gdt, uint32_t a, 
		     double rho, double scale)
{
  double **gdc = _gcounts.data();
  FastQueue &f = _scounts[a];
  for (uint32_t k = 0; k < _k; ++k) {
    gdc[a][k] = (1 - rho) * gdc[a][k] + rho * scale * gdt[a][k];
    if (gdc[a][k] > 1e-05)
      f.update(k, gdc[a][k]);
  }
}

void
FastAMMSparseK::infer()
{
  struct timeval s;
  s.tv_sec = 0;
  s.tv_usec = 0;
  while (1) {
    _lambdat.zero();
    set_dir_exp(_lambda, _Elogbeta);

    struct timeval tv3,tv4,r;
    gettimeofday(&tv3, NULL);
    
    vector<uint32_t> nodes;
    opt_process(nodes);

    gettimeofday(&tv4, NULL);
    timeval_subtract(&r, &tv4, &tv3);
    timeval_add(&s, &r);

    double scale = (double)_n / 2;
    double **gdt = _gammat.data();

    debug("neighbors size = %d\n", (int)nodes.size());
    for (uint32_t i = 0; i < nodes.size(); ++i) 
      debug("%d,", nodes[i]);
    debug("\n");

    for (uint32_t i = 0; i < nodes.size(); ++i) {
      uint32_t a = nodes[i];
      _noderhot[a] = pow(_nodetau0 + _nodec[a], -1 * _nodekappa);

      SparseCounts::iterator j = _scounts.find(a);
      FastQueue &fq = j->second;
      fq.update_counts(a, (const double **)gdt, _noderhot[a], scale);

      //if (a == _start_node) {
      //printf("-> node:%d, Q:%s\n", a, fq.s().c_str());
      //fflush(stdout);
      //}

      _nodec[a]++;
      _iter_map[a]++;
    }

    if (!_env.nolambda)  {
      _rhot = pow(_tau0 + (_iter - _lambda_start_iter + 1), -1 * _kappa);
      
      double **ldt = _lambdat.data();
      double **ed = _eta.data();
      double **ld = _lambda.data();
      
      for (uint32_t k = 0; k < _k; ++k)
	for (uint32_t t = 0; t < _t; ++t) {
	  ldt[k][t] = ed[k][t] + scale  * ldt[k][t];
	  ld[k][t] = (1 - _rhot) * ld[k][t] + _rhot * ldt[k][t];
	}
    }
    
    _iter++;

    fflush(stdout);

    if (_iter % _env.reportfreq == 0) {
      //printf("opt_process took %ld:%ld secs\n",
      //s.tv_sec, s.tv_usec);
      printf("iteration = %d took %d secs (family:%d)\n", _iter, duration(), _family);
      s.tv_sec = 0;
      s.tv_usec = 0;

      //for (uint32_t i = 0; i < _n; ++i) {
      //SparseCounts::iterator j =  _scounts.find(i);
      //if (j != _scounts.end()) {
      //FastQueue &f = _scounts[i];
      //printf("node:%d, f:%s\n", i, f.s().c_str());
      //fflush(stdout);
      //}
      //}

      double n_mean = _neighbors.mean();
      double n_stdev = _neighbors.stdev(n_mean);
      printf("neighbor size = (mean:%.5f, stdev:%.5f)", n_mean, n_stdev);
      fprintf(_cmapf,"%d\t%d\t%.5f\t%.5f\n", _iter, duration(), n_mean, n_stdev);
      fflush(_cmapf);
      
      fflush(stdout);
      
      _neighbors.zero();

      set_dir_exp(_scounts, _Elogpi);
      set_dir_exp(_lambda, _Elogbeta);
      //save_model();
      //compute_and_log_groups();
      
      double a, b, c;
      heldout_likelihood(a, b, c);
      
      //validation_likelihood(a, b, c);
      //training_likelihood(a, b, c);

#ifdef GLOBALPHIS
      approx_log_likelihood();
#endif
    }
  }
}

void
FastAMMSparseK::save_model()
{
  FILE *gammaf = fopen(Env::file_str("/gamma.txt").c_str(), "w");
  const double ** const gd = _gamma.const_data();
  for (uint32_t i = 0; i < _n; ++i) {
    const IDMap &m = _network.seq2id();
    IDMap::const_iterator idt = m.find(i);
    if (idt != m.end()) {
      fprintf(gammaf,"%d\t", i);
      debug("looking up i %d\n", i);
      fprintf(gammaf,"%d\t", (*idt).second);
      for (uint32_t k = 0; k < _k; ++k) {
	if (k == _k - 1)
	  fprintf(gammaf,"%.5f\n", gd[i][k]);
	else
	  fprintf(gammaf,"%.5f\t", gd[i][k]);
      }
    }
  }
  fclose(gammaf);

  FILE *lambdaf = fopen(Env::file_str("/lambda.txt").c_str(), "w");
  const double ** const ld = _lambda.const_data();
  for (uint32_t k = 0; k < _k; ++k) {
    fprintf(lambdaf,"%d\t", k);
    for (uint32_t t = 0; t < _t; ++t) {
      if (t == _t - 1)
	fprintf(lambdaf,"%.5f\n", ld[k][t]);
      else
	fprintf(lambdaf,"%.5f\t", ld[k][t]);
    }
  }
  fclose(lambdaf);
}

void
FastAMMSparseK::compute_and_log_groups()
{
  FILE *groupsf = fopen(Env::file_str("/groups.txt").c_str(), "w");
  FILE *summaryf = fopen(Env::file_str("/summary.txt").c_str(), "a");
  FILE *commf = fopen(Env::file_str("/communities.txt").c_str(), "w");
  MapVec communities;

  estimate_all_pi();
  uint32_t unlikely = 0;

  char buf[32];
  const IDMap &seq2id = _network.seq2id();
  ostringstream sa;
  Array groups(_n);
  Array pi_i(_k);
  for (uint32_t i = 0; i < _n; ++i) {
    sa << i << "\t";
    IDMap::const_iterator it = seq2id.find(i);
    uint32_t id = 0;
    if (it == seq2id.end()) { // single node
      id = i;
    } else
      id = (*it).second;

    sa << id << "\t";
    _Epi.slice(0, i, pi_i);
    //estimate_pi(i, pi_i);
    double max = .0;
    for (uint32_t j = 0; j < _k; ++j) {
      memset(buf, 0, 32);
      sprintf(buf,"%.3f", pi_i[j]);
      sa << buf << "\t";
      if (pi_i[j] > max) {
	max = pi_i[j];
	groups[i] = j;
      }
    }
    Array pi_m(_k);
    for (uint32_t m = 0; m < _n; ++m) {
      if (i < m) { 
	yval_t y = get_y(i,m);

	if (y == 1) {
	  Array beta(_k);
	  estimate_beta(beta);
	  _Epi.slice(0, m, pi_m);
	  uint32_t max_k = 65535;
	  double max = inner_prod_max(pi_i, pi_m, beta, max_k);
	  if (max < 0.9) {
	    unlikely++;
	    continue;
	  }
	  assert (max_k < _k);
	  communities[max_k].push_back(i);
	  communities[max_k].push_back(m);
	}
      }
    }
    sa << groups[i] << "\n";
  }
  printf("unlikely = %d\n", unlikely);
  fflush(stdout);
  fprintf(groupsf, "%s", sa.str().c_str());

  D1Array<int> s(_k);
  std::map<uint32_t, vector<uint32_t> > comm;
  for (uint32_t i = 0; i < _n; ++i) {
    s[groups[i]]++;
    comm[groups[i]].push_back(i);
  }
  for (uint32_t i = 0; i < _k; ++i)
    fprintf(summaryf, "%d\t", s[i]);
  fprintf(summaryf, ":%d\n", unlikely);

  for (std::map<uint32_t, vector<uint32_t> >::const_iterator i = communities.begin();
       i != communities.end(); ++i) {
    //fprintf(commf, "%d\t", i->first);
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

  fprintf(summaryf,"\n");
  fflush(groupsf);
  fflush(summaryf);
  fflush(commf);
  fclose(groupsf);
  fclose(summaryf);
  fclose(commf);
  
  if (_env.nmi) {
    if (!_env.benchmark) {
      char cmd[1024];
      sprintf(cmd, "/usr/local/bin/mutual %s %s >> %s", 
	      _env.ground_truth_fname.c_str(),
	      Env::file_str("/communities.txt").c_str(), 
	      Env::file_str("/mutual.txt").c_str());
      if (system(cmd) < 0)
	lerr("error spawning cmd %s:%s", cmd, strerror(errno));
    } else {
      char cmd[1024];
      sprintf(cmd, "/usr/local/bin/mutual %s %s >> %s", 
	      Env::file_str("/ground_truth.txt").c_str(),
	      Env::file_str("/communities.txt").c_str(), 
	      Env::file_str("/mutual.txt").c_str());
      if (system(cmd) < 0)
	lerr("error spawning cmd %s:%s", cmd, strerror(errno));
    }
  }
}

void
FastAMMSparseK::update_cache(uint32_t node)
{
  SparseCounts::iterator i = _scounts.find(node);
  FastQueue &fq = i->second;
  fq.update_cache();
}

void
FastAMMSparseK::opt_process(vector<uint32_t> &nodes)
{
  _start_node = gsl_rng_uniform_int(_r, _n);
  double **ldt = _lambdat.data();
  
  // full inference only around the neighborhood _start_node
  set_dir_exp(_start_node, _scounts, _Elogpi);
  _gammat.zero(_start_node);
  
  //nodes.push_back(_start_node);

  const vector<uint32_t> *edges = _network.get_edges(_start_node);
  bool singleton = false;
  if (!edges)  // singleton node
    singleton = true;

  struct timeval s;
  s.tv_sec = 0;
  s.tv_usec = 0;

  uint32_t l_size = 0;
  if (!singleton) {
    for (uint32_t i = 0; i < edges->size(); ++i) {
      uint32_t a = (*edges)[i];
      
      Edge e(_start_node,a);
      Network::order_edge(_env, e);
      const SampleMap::const_iterator u1 = _heldout_map.find(e);
      if (u1 != _heldout_map.end())
	continue;

      const SampleMap::const_iterator u2 = _validation_map.find(e);
      if (u2 != _validation_map.end())
	continue;      

      l_size++;

      nodes.push_back(a);
      set_dir_exp(a, _scounts, _Elogpi);
      _gammat.zero(a);

      uint32_t p = e.first;
      uint32_t q = e.second;

      assert (p != q);
      yval_t y = get_y(p, q);

      mark_seen(p,q);
      assert (y == 1);

      struct timeval tv3,tv4,r;
      gettimeofday(&tv3, NULL);

      _pcomp.reset(p,q,y);
      _pcomp.update_phis_until_conv();

      gettimeofday(&tv4, NULL);
      timeval_subtract(&r, &tv4, &tv3);
      timeval_add(&s, &r);

      const Array &phi1 = _pcomp.phi1();
      const Array &phi2 = _pcomp.phi2();

      _gammat.add_slice(p, phi1);
      _gammat.add_slice(q, phi2);

      for (uint32_t k = 0; k < _k; ++k)
	for (uint32_t t = 0; t < _t; ++t)
	  ldt[k][t] += phi1[k] * phi2[k] * (t == 0 ? y : (1-y));
    }
  }

  //printf("time for %ld edge updates = %ld:%ld\n", edges->size(), 
  //s.tv_sec, s.tv_usec);

  s.tv_sec = 0;
  s.tv_usec = 0;

  vector<uint32_t> neighbors;
  get_similar_nodes2(_start_node, neighbors);

  uint32_t nl_inf_size = 0;
  for (uint32_t i = 0; i < neighbors.size(); ++i) {
    uint32_t a = neighbors[i];
    assert (a != _start_node);

    yval_t y = get_y(_start_node, a);
    if (y == 1)  // already processed links
      continue;
    
    Edge e(_start_node,a);
    Network::order_edge(_env, e);
    const SampleMap::const_iterator u1 = _heldout_map.find(e);
    if (u1 != _heldout_map.end())
      continue;
    
    const SampleMap::const_iterator u2 = _validation_map.find(e);
    if (u2 != _validation_map.end())
      continue;      

    set_dir_exp(a, _scounts, _Elogpi);
    nodes.push_back(a);
    _gammat.zero(a);
    
    uint32_t p = e.first;
    uint32_t q = e.second;

    mark_seen(p,q);
    nl_inf_size++;

    struct timeval tv3,tv4,r;
    gettimeofday(&tv3, NULL);
    
    _pcomp.reset(p,q,y);
    _pcomp.update_phis_until_conv();

    gettimeofday(&tv4, NULL);
    timeval_subtract(&r, &tv4, &tv3);
    timeval_add(&s, &r);

    const Array &phi1 = _pcomp.phi1();
    const Array &phi2 = _pcomp.phi2();

    _gammat.add_slice(p, phi1);
    _gammat.add_slice(q, phi2);

    for (uint32_t k = 0; k < _k; ++k)
      for (uint32_t t = 0; t < _t; ++t)
	ldt[k][t] += phi1[k] * phi2[k] * (t == 0 ? y : (1-y));
  }
  
  //if (nl_inf_size)
  //printf("time for %u non-edge updates = %ld:%ld\n", nl_inf_size, 
  //s.tv_sec, s.tv_usec);

  _neighbors[_iter % _env.reportfreq] = nl_inf_size;
  
  //uint32_t nl_ninf_size = (_n - 1) - (l_size + nl_inf_size);
}

#ifdef GLOBALPHIS
double
FastAMMSparseK::approx_log_likelihood()
{
  const double ** const etad = _eta.const_data();
  const double * const alphad = _alpha.const_data();
  const double ** const elogbetad = _Elogbeta.const_data();
  const double ** const elogpid = _Elogpi.const_data();
  const double ** const ld = _lambda.const_data();
  const double ** const gd = _gamma.const_data();
  const double *** const gphi1d = FPhiComp2::gphi1().const_data();
  const double *** const gphi2d = FPhiComp2::gphi2().const_data();

#ifndef SPARSE_NETWORK
  const yval_t ** const yd = _network.y().const_data();
#endif

  double s = .0, v = .0;
  for (uint32_t k = 0; k < _k; ++k) {
    v = .0;
    for (uint32_t t = 0; t < _t; ++t)
      v += gsl_sf_lngamma(etad[k][t]);
    s += gsl_sf_lngamma(_eta.sum(k)) - v;

    v = .0;
    for (uint32_t t = 0; t < _t; ++t)
      v += (etad[k][t] - 1) * elogbetad[k][t];
    s += v;

    v = .0;
    for (uint32_t t = 0; t < _t; ++t)
      v += gsl_sf_lngamma(ld[k][t]);
    s -= gsl_sf_lngamma(_lambda.sum(k)) - v;

    v = .0;
    for (uint32_t t = 0; t < _t; ++t)
      v += (ld[k][t] - 1) * elogbetad[k][t];
    s -= v;
  }

  for (uint32_t p = 0; p < _n; ++p) {
    for (uint32_t q = 0; q < _n; ++q) {
      if (p >= q)
	continue;

      Edge e(p,q);
      Network::order_edge(_env, e);
      const SampleMap::const_iterator u = _heldout_map.find(e);
      if (u != _heldout_map.end())
	continue;

      yval_t y = get_y(p,q);
      const double * const phi1 = gphi1d[p][q];
      const double * const phi2 = gphi2d[p][q];

      Array Elogf(_k);
      FPhiComp2::compute_Elogf(p,q,y,_k,_t,_Elogbeta,Elogf);
      double *elogfd = Elogf.data();

      for (uint32_t k = 0; k < _k; ++k)
	s += phi1[k] * phi2[k] * elogfd[k];

      if (y == 1)
	for (uint32_t g = 0; g < _k; ++g)
	  for (uint32_t h = 0; h < _k; ++h)
	    if (g != h)
	      s += phi1[g] * phi2[h] * _env.logepsilon;

      for (uint32_t k = 0; k < _k; ++k) {
	s += phi1[k] * elogpid[p][k];
	s += phi2[k] * elogpid[q][k];
      }

      for (uint32_t k = 0; k < _k; ++k) {
	s -= log(phi1[k]) * phi1[k];
	s -= log(phi2[k]) * phi2[k];
      }
    }
  }

  for (uint32_t p = 0; p < _n; ++p) {
    v = .0;
    for (uint32_t k = 0; k < _k; ++k)
      v += gsl_sf_lngamma(alphad[k]);
    s += gsl_sf_lngamma(_alpha.sum()) - v;

    v = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      v += (alphad[k] - 1) * elogpid[p][k];
    }
    s += v;

    v = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      double qq = gd[p][k];
      if (gd[p][k] < 1e-30)
	qq = 1e-30;
      v += gsl_sf_lngamma(qq);
    }
    s -= gsl_sf_lngamma(_gamma.sum(p)) - v;

    v = .0;
    for (uint32_t k = 0; k < _k; ++k)
      v += (gd[p][k] - 1) * elogpid[p][k];
    s -= v;
  }

  printf("approx. log likelihood = %f\n", s);
  fprintf(_lf, "%d\t%d\t%.5f\n", _iter, duration(), s);
  fflush(_lf);

  double thresh = 0.00001;
  double w = s;
  if (_env.accuracy && _iter > 1000) {
    if (w > _prev_w && _prev_w != 0 && fabs((w - _prev_w) / _prev_w) < thresh) {
      FILE *f = fopen(Env::file_str("/done.txt").c_str(), "w");
      fprintf(f, "%d\t%d\t%.5f\n", _iter, duration(), w); 
      fclose(f);
      if (_env.nmi) {
	char cmd[1024];
	sprintf(cmd, "/usr/local/bin/mutual %s %s >> %s", 
		Env::file_str("/ground_truth.txt").c_str(),
		Env::file_str("/communities.txt").c_str(), 
		Env::file_str("/done.txt").c_str());
	system(cmd);
      }
      //exit(0);
    }
  }
  _prev_w = w;
  return s;
}
#endif

void
FastAMMSparseK::heldout_likelihood(double &a, double &a0, double &a1)
{
  if (_env.accuracy)
    return;
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
    bool seen = false; // TODO: fix heldout for sparse network
#endif

    assert (!seen);
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
    debug("edge likelihood for (%d,%d) is %f\n", p,q,u);
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
  if (_iter > 5000) {
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
  //if (stop)
  //exit(0);
}

void
FastAMMSparseK::validation_likelihood(double &av, double &av0, double &av1)
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
    bool seen = false;
#endif
    
    assert (!seen);

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
    debug("edge likelihood for (%d,%d) is %f\n", p,q,u);
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
FastAMMSparseK::training_likelihood(double &av, double &av0, double &av1)
{
  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  uint32_t c = 0;
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

    if (!seen)
      c++;
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
    debug("edge likelihood for (%d,%d) is %f\n", p,q,u);
  }
  fprintf(_trf, "%d\t%d\t%.5f\t%d\t%.5f\t%d\t%.5f\t%d\t%d\n",
	  _iter, duration(), s / k, k,
	  szeros / kzeros, kzeros, sones / kones, kones,c);
  fflush(_trf);

  av = s / k;
  av0 = szeros / kzeros;
  av1 = sones / kones;

  double thresh = _env.stopthresh;
  double w = av;
  bool stop = false;
  if (_env.accuracy && _iter > 1000 && !_training_done) {
    if (w > _prev_t && _prev_t != 0 && fabs((w - _prev_t) / _prev_t) < thresh) {
      stop = true;
    } else if (w > _max_t) {
      _max_t = w;
      _nt = 0;
    } else if (w < _max_t) {
      _nt++;
    }
    if (_nt > 3)
      stop = true;
  }

  if (stop) {
    FILE *f = fopen(Env::file_str("/donet.txt").c_str(), "w");
    fprintf(f, "%d\t%d\t%.5f\n", _iter, duration(), w); 
    fclose(f);
    
    if (_env.nmi) {
      char cmd[1024];
	sprintf(cmd, "/usr/local/bin/mutual %s %s >> %s", 
		Env::file_str("/ground_truth.txt").c_str(),
		Env::file_str("/communities.txt").c_str(), 
		Env::file_str("/donet.txt").c_str());
	if (system(cmd) < 0)
	  lerr("error spawning cmd %s:%s", cmd, strerror(errno));
    }
    //exit(0);
    _training_done = true;
  }
  _prev_w = w;
}



void
FastAMMSparseK::moving_heldout_likelihood(EdgeList &sample)
{
  if (_env.accuracy)
    return;

  uint32_t p, q;
  double lones = .0, lzeros = .0;
  uint32_t kones = 0, kzeros = 0;
  double s = .0;
  uint32_t k = 0;
  for (EdgeList::const_iterator i = sample.begin(); i != sample.end(); ++i) {
    p = i->first;
    q = i->second;
    assert (p != q);

#ifndef SPARSE_NETWORK
    const yval_t ** const yd = _network.y().const_data();
    yval_t y = yd[p][q] & 0x01;
    bool seen = yd[p][q] & 0x80;
#else
    yval_t y = _network.y(p,q);
    bool seen = false; // TODO sparse network heldout
#endif

    if (not seen) {
      double l = .0;
      l = edge_likelihood(p,q,y);
      s += l;
      if (y == 1) {
	lones += l;
	kones++;
      } else {
	lzeros += l;
	kzeros++;
      }
      k += 1;
    }
  }

  double h = .0;
  if (k > 0)
    h = s / k;
  else
    h = _env.illegal_likelihood;
  double hones = .0;
  if (kones > 0)
    hones = lones / kones;
  else
    hones = _env.illegal_likelihood;
  double hzeros = .0;
  if (kzeros > 0)
    hzeros = lzeros / kzeros;
  else
    hzeros = _env.illegal_likelihood;

  if (h != _env.illegal_likelihood) {
    fprintf(_statsf, "%d\t%d\t%.5f\t%.5f\t%.5f\t%d\t%d\n",
	    _iter, duration(), h, hones, hzeros, kones, kzeros);
    fflush(_statsf);
  }
}

int
FastAMMSparseK::load_model()
{
  fprintf(stderr, "+ loading gamma\n");
  double **gd = _gamma.data();
  FILE *gammaf = fopen("gamma.txt", "r");
  if (!gammaf)
    return -1;
  uint32_t n = 0;
  int sz = 32*_k;
  char *line = (char *)malloc(sz);
  while (!feof(gammaf)) {
    if (fgets(line, sz, gammaf) == NULL)
      break;
    //assert (fscanf(gammaf, "%[^\n]", line) > 0);
    debug("line = %s\n", line);
    uint32_t k = 0;
    char *p = line;
    do {
      char *q = NULL;
      double d = strtod(p, &q);
      if (q == p) {
	if (k < _k - 1) {
	  fprintf(stderr, "error parsing gamma file\n");
	  assert(0);
	}
	break;
      }
      p = q;
      if (k >= 2) // skip node id and seq
	gd[n][k-2] = d;
      k++;
    } while (p != NULL);
    n++;
    debug("read %d lines\n", n);
    memset(line, 0, sz);
  }
  assert (n == _n);
  fclose(gammaf);
  memset(line, 0, sz);

  fprintf(stderr, "+ loading lambda\n");
  double **ld = _lambda.data();
  FILE *lambdaf = fopen("lambda.txt", "r");
  if (!lambdaf)
    return -1;
  uint32_t k = 0;
  while (!feof(lambdaf)) {
    if (fgets(line, sz, lambdaf) == NULL)
      break;
    debug("line = %s\n", line);
    uint32_t t = 0;
    char *p = line;
    do {
      char *q = NULL;
      double d = strtod(p, &q);
      if (q == p) {
	if (t < _t - 1) {
	  fprintf(stderr, "error parsing gamma file\n");
	  assert(0);
	}
	break;
      }
      p = q;
      if (t >= 1) // skip seq
	ld[k][t-1] = d;
      t++;
    } while (p != NULL);
    k++;
    debug("read %d lines\n", n);
    memset(line, 0, sz);
  }
  assert (k == _k);
  fclose(lambdaf);
  free(line);

  return 0;
}

