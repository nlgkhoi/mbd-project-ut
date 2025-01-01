# Setup PySpark on Mac Silicon
## Commands
```
brew install openjdk@11
# the version should be 11 for pyspark to work
java -version 

# Note: setup ENV variables of JAVA as in the below section before proceeding
source ~/.zshrc

# Install Apache Spark
brew install apache-spark

# Note: Setup other ENV varialbes of Spark as in the below section before proceeding
source ~/.zshrc

# Install Hadoop
brew install hadoop
# check hadoop installation
hadoop version
# for me it outputs something like "Hadoop 3.4.1 ..."

# Install hadoop 
# Install PySpark
pip install pyspark

# Verify the installation
spark-submit wordcount.py

```

## `~/.bashrc` or `~/.zshrc` ENV variables:
```
# Java
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@11/11.0.25/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH

# Java Security Options
export SPARK_OPTS="--driver-java-options=-Djava.security.auth.login.config=/dev/null"
export SPARK_LOCAL_IP=127.0.0.1

# Spark
export SPARK_HOME=/opt/homebrew/Cellar/apache-spark/3.5.4/libexec
export PATH=$SPARK_HOME/bin:$PATH

# Python for PySpark
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# Hadoop
export HADOOP_HOME=/opt/homebrew/Cellar/hadoop/3.4.1/libexec
export PATH=$HADOOP_HOME/bin:$PATH
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=$HADOOP_HOME/lib/native"
```