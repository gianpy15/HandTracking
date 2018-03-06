<?php
    include 'local_paths.php';
    $script = "source/php_called_scripts/get_frame.py";
    $args = "2>&1";
    $cmd = $python_interpreter." ".$script_base.$script." ".$args;
    $out = exec($cmd, $rets, $errorcode);
    echo $out;
?>