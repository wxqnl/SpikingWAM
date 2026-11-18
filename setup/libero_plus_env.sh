# Source after setup/env.sh. Select LIBERO-Plus ahead of LIBERO on the module search path.
export SW_LIBERO_PLUS="${SW_LIBERO_PLUS:-$SW_BENCH/LIBERO-plus}"
export LIBERO_CONFIG_PATH="${SW_LIBERO_PLUS_CONFIG:-$SW_LIBERO_PLUS/.libero}"
export PYTHONPATH="$SW_ROOT:$SW_LIBERO_PLUS:$SW_BENCH/libero_plus_pydeps${PYTHONPATH:+:$PYTHONPATH}"
if [ -d "$SW_BENCH/libero_plus_magick" ]; then
  export MAGICK_HOME="$SW_BENCH/libero_plus_magick"
  export LD_PRELOAD="$MAGICK_HOME/lib/libglib-2.0.so.0${LD_PRELOAD:+:$LD_PRELOAD}"
fi
