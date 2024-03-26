from adapter.STSRealtimeAdapter import STSRealtimeAdapter

if __name__ == "__main__":
    try:
        adapter = STSRealtimeAdapter()
        adapter.listen()
        # adapter.sounddevice()
    except Exception as e:
        print(e)