#!/usr/bin/env python3
"""
Pub/Sub Topic and Subscription Setup

This script creates the necessary Pub/Sub topics and subscriptions for the IoT anomaly detection system.
"""

import argparse
from google.cloud import pubsub_v1
from google.api_core import exceptions

def create_topic(project_id: str, topic_name: str) -> None:
    """Create a Pub/Sub topic if it doesn't exist."""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_name)
    
    try:
        topic = publisher.create_topic(request={"name": topic_path})
        print(f"Created topic: {topic.name}")
    except exceptions.AlreadyExists:
        print(f"Topic {topic_name} already exists.")

def create_subscription(project_id: str, topic_name: str, subscription_name: str) -> None:
    """Create a Pub/Sub subscription if it doesn't exist."""
    subscriber = pubsub_v1.SubscriberClient()
    publisher = pubsub_v1.PublisherClient()
    
    topic_path = publisher.topic_path(project_id, topic_name)
    subscription_path = subscriber.subscription_path(project_id, subscription_name)
    
    try:
        subscription = subscriber.create_subscription(
            request={"name": subscription_path, "topic": topic_path}
        )
        print(f"Created subscription: {subscription.name}")
    except exceptions.AlreadyExists:
        print(f"Subscription {subscription_name} already exists.")

def main():
    """Main function to set up Pub/Sub resources."""
    parser = argparse.ArgumentParser(description='Setup Pub/Sub topics and subscriptions')
    parser.add_argument('--project-id', required=True, help='Google Cloud Project ID')
    
    args = parser.parse_args()
    
    # Create topics
    topics = [
        'iot-sensor-data',
        'anomaly-alerts',
        'processed-data'
    ]
    
    for topic in topics:
        create_topic(args.project_id, topic)
    
    # Create subscriptions
    subscriptions = [
        ('iot-sensor-data', 'dataflow-subscription'),
        ('anomaly-alerts', 'alert-handler-subscription'),
        ('processed-data', 'bigquery-subscription')
    ]
    
    for topic, subscription in subscriptions:
        create_subscription(args.project_id, topic, subscription)
    
    print("\nPub/Sub setup completed!")
    print("\nTopics created:")
    for topic in topics:
        print(f"  - {topic}")
    
    print("\nSubscriptions created:")
    for topic, subscription in subscriptions:
        print(f"  - {subscription} (for topic: {topic})")

if __name__ == "__main__":
    main()

