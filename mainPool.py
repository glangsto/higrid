from multiprocessing import Pool

# Prepare shared config object
class Config:
    pass

cfg = Config()
cfg.velocity = args.velocity
cfg.rest_freq = args.rest_freq
cfg.nchan = args.nchan
cfg.ra_grid = ra_grid
cfg.dec_grid = dec_grid
cfg.beam_sigma = beam_sigma

# Build job list
job_args = [
    (fname, freq_axis_topo, vel_axis_kms, cfg)
    for fname in args.files
]

# Run in parallel
with Pool() as pool:
    results = pool.map(process_spectrum, job_args)

# Accumulate results
for cube_part, weight_part in results:
    cube_data += cube_part
    weight_data += weight_part
