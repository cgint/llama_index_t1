#!/bin/bash


docker run \
    --publish=7474:7474 \
    --publish=7687:7687 \
    --volume=./neo4j_data:/data \
    --volume=./neo4j_plugins:/plugins \
    --net host \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
    -e NEO4J_dbms_security_procedures_unrestricted=apoc.\\\* \
    neo4j