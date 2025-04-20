# Node-RED for Pulse Rate Monitoring

## Introduction
Node-RED is a flow-based programming tool that can be used to create visual workflows for your pulse rate monitoring system. It provides a browser-based editor that makes it easy to wire together flows using a wide range of nodes.

## Starting Node-RED
To start Node-RED, open a command prompt and run:
```
node-red
```

Once started, you can access the Node-RED editor by opening a browser and navigating to:
```
http://localhost:1880
```

## Potential Uses with MAX30102 Sensor

1. **Data Visualization**: Create dashboards to visualize real-time pulse rate data
2. **Data Processing**: Process and filter sensor data before analysis
3. **Integration**: Connect your pulse rate monitoring system with other services or IoT devices
4. **Alerts**: Set up notifications when pulse rates exceed certain thresholds

## Example Flow for Pulse Rate Monitoring

You can create a flow that:
1. Receives serial data from the Arduino with MAX30102 sensor
2. Processes the IR and Red values
3. Calculates pulse rate
4. Displays the data on a dashboard
5. Stores historical data

## Useful Nodes for this Project

- `serial in` - To receive data from Arduino
- `function` - To process data and calculate pulse rate
- `dashboard` - To create visual representations
- `file` - To store data for later analysis

## Making Node-RED Start on Boot

To make Node-RED start automatically when your computer boots:

### Windows
Create a shortcut to Node-RED in your startup folder:
1. Press `Win+R` and type `shell:startup`
2. Create a new shortcut
3. Set the target to: `node-red`

## Additional Resources

- [Node-RED Documentation](https://nodered.org/docs/)
- [Node-RED Dashboard](https://flows.nodered.org/node/node-red-dashboard)
- [Serial Port Nodes](https://flows.nodered.org/node/node-red-node-serialport)