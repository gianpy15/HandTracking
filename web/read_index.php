<?php
    include 'local_paths.php';
    $script = "source/php_called_scripts/get_index_content.py";
    $args = $_POST["framename"]." 2>&1";
    $cmd = $python_interpreter." ".$script_base.$script." ".$args;
    $out = exec($cmd, $rets, $errorcode);
    echo $out;
?>