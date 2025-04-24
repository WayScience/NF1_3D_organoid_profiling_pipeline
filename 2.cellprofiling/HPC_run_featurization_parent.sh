#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=parent_featurize-%j.out

module load miniforge
conda init bash
conda activate GFF_featurization

well_fov=$1
use_GPU=$2


echo "Submitting jobs for $well_fov"
echo "Using GPU: $use_GPU"

number_of_jobs=$(squeue -u $USER | wc -l)
while [ $number_of_jobs -gt 990 ]; do
    sleep 1s
    number_of_jobs=$(squeue -u $USER | wc -l)
done

cd slurm_scripts || exit

if [ "$use_GPU" = "TRUE" ]; then
    echo "Running GPU version"

    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=10:00 \
        --output=area_shape_gpu_child-%j.out \
        run_area_shape_child.sh $well_fov $use_GPU

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=30:00 \
        --output=colocalization_gpu_child-%j.out \
        run_colocalization_child.sh $well_fov $use_GPU

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=1:30:00 \
        --output=granularity_gpu_child-%j.out \
        run_granularity_child.sh $well_fov $use_GPU

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=3:00:00 \
        --output=intensity_gpu_child-%j.out \
        run_intensity_child.sh $well_fov $use_GPU

else
    echo "Running CPU version"

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=10:00 \
        --output=area_shape_cpu_child-%j.out \
        run_area_shape_child.sh $well_fov $use_GPU

    sbatch \
        --nodes=1 \
        --ntasks=20 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=1:00:00 \
        --output=colocalization_cpu_child-%j.out \
        run_colocalization_child.sh $well_fov $use_GPU

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=6:00:00 \
        --output=granularity_cpu_child-%j.out \
        run_granularity_child.sh $well_fov $use_GPU

    sbatch \
        --nodes=1 \
        --ntasks=20 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=6:00:00 \
        --output=intensity_cpu_child-%j.out \
        run_intensity_child.sh $well_fov $use_GPU

fi

sbatch \
    --nodes=1 \
    --ntasks=1 \
    --partition=amilan \
    --qos=normal \
    --account=amc-general \
    --time=10:00 \
    --output=neighbors_child-%j.out \
    run_neighbors_child.sh $well_fov $use_GPU



sbatch \
    --nodes=1 \
    --ntasks=20 \
    --partition=amilan \
    --qos=normal \
    --account=amc-general \
    --time=16:00:00 \
    --output=texture_child-%j.out \
    run_texture_child.sh $well_fov $use_GPU


cd ../ || exit

echo "Featurization done"

