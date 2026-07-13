import logging

import dbof.cutout_dataset_creation.config as config
import dbof.cutout_dataset_creation.processing as processing
from dbof.utils.logging import generate_logging


def main():
    """
    Entry point for native-grid LLC cutout dataset generation.

    Parses arguments, sets up logging, and runs the cutout pipeline.
    """
    cli = config.parse_args()
    cfg = config.load_config(cli.config)
    print(cfg)

    # override run_id if passed in through cli
    if cli.run_id is not None:
        cfg = config.JobConfig(
            run=config.RunConfig(run_id=cli.run_id, log_dir=cfg.run.log_dir),
            input=cfg.input,
            sampling=cfg.sampling,
            output=cfg.output,
            features=cfg.features,
            runtime=cfg.runtime,
        )

    generate_logging(cfg.run, log_filename="generate_cutout_dataset.log")

    logging.info("Arguments parsed successfully. Logging set up. Running script.")

    processing.run(cfg)


if __name__ == "__main__":
    main()
